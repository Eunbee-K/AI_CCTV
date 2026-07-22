import { api } from "./api.js";
import { selectVideo, showDetectionFrame, getCurrentVideo, setDetections, onDetectionPass } from "./player.js";
import {
  rowVisible, onFilterChange, setKnownClasses, getKnownClasses,
  isClassEnabled, toggleClass, setConfMin,
} from "./filters.js";

const tbody = document.getElementById("resultsBody");
const classFilterEl = document.getElementById("classFilter");
const confSlider = document.getElementById("confSlider");
const confVal = document.getElementById("confVal");

const openGroups = new Set(); // "video|dist" keys currently expanded
const selected = new Set(); // "video|time_s" keys currently selected for deletion

let lastData = null; // 마지막 결과 (필터 재적용/통계용)

function rowKey(video, time_s) {
  return `${video}|${time_s}`;
}

function editableFieldForColumn(colIdx) {
  // 0:순번 1:시간 2:관로번호 3:직경 4:거리 5:결함항목 6:비고 7:오탐
  return ["", "", "pipe_id", "dia", "dist", "defects", "note", ""][colIdx] || "";
}

function cellValue(row, colIdx) {
  switch (colIdx) {
    case 0: return row.seq;
    case 1: return row.time_str;
    case 2: return row.pipe_id;
    case 3: return row.dia;
    case 4: return row.dist;
    case 5: return row.defects.join(", ");
    case 6: return row.note;
    default: return "";
  }
}

async function commitEdit(row, colIdx, newValue) {
  const field = editableFieldForColumn(colIdx);
  if (!field) return;
  await api.editRow(row.filename, row.time_s, field, newValue);
  await refreshResults();
}

function makeCellEditable(td, row, colIdx) {
  td.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    if (colIdx === 0 || colIdx === 1 || colIdx === 7) return; // 순번/시간/오탐은 편집 불가
    if (td.querySelector("input")) return;
    const original = cellValue(row, colIdx);
    td.textContent = "";
    const input = document.createElement("input");
    input.className = "cell-edit";
    input.value = original;
    td.appendChild(input);
    input.focus();
    input.select();

    const save = () => {
      const val = input.value;
      commitEdit(row, colIdx, val);
    };
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") input.blur();
      if (ev.key === "Escape") {
        input.removeEventListener("blur", save);
        refreshResults();
      }
    });
    input.addEventListener("blur", save, { once: true });
  });
}

function activateRow(row) {
  if (row.filename !== getCurrentVideo()) {
    selectVideo(row.filename).then(() => showDetectionFrame(row.filename, row.time_s, row.boxes));
  } else {
    showDetectionFrame(row.filename, row.time_s, row.boxes);
  }
}

function buildRowTr(row, extraClass) {
  const tr = document.createElement("tr");
  tr.className = `row-child ${extraClass || ""}`;
  tr.dataset.video = row.filename;
  tr.dataset.timeS = row.time_s;
  if (row.fp) tr.classList.add("row-fp");
  if (!rowVisible(row)) tr.classList.add("row-filtered");

  const key = rowKey(row.filename, row.time_s);
  if (selected.has(key)) tr.classList.add("row-selected");

  const cols = [row.seq, row.time_str, row.pipe_id, row.dia, row.dist, row.defects.join(", "), row.note];
  cols.forEach((val, idx) => {
    const td = document.createElement("td");
    td.textContent = val;
    if (idx === 5) td.classList.add("col-defects");
    makeCellEditable(td, row, idx);
    tr.appendChild(td);
  });

  // 오탐 체크박스 (체크 = 오탐 → 보고서/통계에서 제외)
  const tdFp = document.createElement("td");
  tdFp.className = "col-fp";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = !!row.fp;
  cb.title = "오탐으로 표시 (보고서/통계에서 제외)";
  cb.addEventListener("click", (e) => e.stopPropagation());
  cb.addEventListener("change", async () => {
    await api.editRow(row.filename, row.time_s, "fp", cb.checked);
    await refreshResults();
  });
  tdFp.appendChild(cb);
  tr.appendChild(tdFp);

  tr.addEventListener("click", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (!e.ctrlKey && !e.metaKey) selected.clear();
    if (selected.has(key)) selected.delete(key);
    else selected.add(key);
    syncSelectionClasses();
    activateRow(row);
  });

  return tr;
}

function syncSelectionClasses() {
  tbody.querySelectorAll("tr[data-time-s]").forEach((tr) => {
    const key = rowKey(tr.dataset.video, Number(tr.dataset.timeS));
    tr.classList.toggle("row-selected", selected.has(key));
  });
  tbody.querySelectorAll("tr.row-group-parent").forEach((tr) => {
    tr.classList.toggle("row-selected", selected.has(`group:${tr.dataset.groupKey}`));
  });
}

// ───────── 필터 UI (신뢰도 슬라이더 + 클래스 칩) ─────────

confSlider.addEventListener("input", () => {
  confVal.textContent = confSlider.value;
  setConfMin(Number(confSlider.value) / 100);
});

function renderClassChips() {
  classFilterEl.innerHTML = "";
  for (const cls of getKnownClasses()) {
    const chip = document.createElement("button");
    chip.className = "class-chip" + (isClassEnabled(cls) ? " on" : "");
    chip.textContent = cls;
    chip.title = "클릭해서 이 결함 종류 표시/숨김";
    chip.addEventListener("click", () => toggleClass(cls));
    classFilterEl.appendChild(chip);
  }
}

onFilterChange(() => {
  renderClassChips();
  if (lastData) renderResults(lastData);
});

// ───────── 렌더링 ─────────

function collectRows(data) {
  const rows = [];
  for (const item of data.display) {
    if (item.type === "row") rows.push(item);
    else if (item.type === "group") rows.push(...item.children);
  }
  return rows;
}

export function renderResults(data) {
  lastData = data;

  // 클래스 필터 후보 + 플레이어 오버레이용 탐지 맵 갱신
  const allRows = collectRows(data);
  const classes = new Set();
  const detMap = {};
  for (const r of allRows) {
    for (const d of r.defects) classes.add(d);
    if (r.boxes && r.boxes.length) {
      (detMap[r.filename] = detMap[r.filename] || {})[r.time_s] = r.fp ? [] : r.boxes;
    }
  }
  setKnownClasses([...classes].sort());
  renderClassChips();
  setDetections(detMap);

  tbody.innerHTML = "";
  let seqParity = 0;

  for (const item of data.display) {
    if (item.type === "separator") {
      const tr = document.createElement("tr");
      tr.className = "row-separator";
      const td = document.createElement("td");
      td.colSpan = 8;
      td.textContent = `[${item.filename}]`;
      tr.appendChild(td);
      tbody.appendChild(tr);
      seqParity = 0;
      continue;
    }

    if (item.type === "row") {
      seqParity += 1;
      const tr = buildRowTr(item, seqParity % 2 === 1 ? "row-odd" : "");
      tbody.appendChild(tr);
      continue;
    }

    if (item.type === "group") {
      seqParity += 1;
      const groupKey = `${item.filename}|${item.dist}`;
      const isOpen = openGroups.has(groupKey);

      const tr = document.createElement("tr");
      tr.className = "row-group-parent";
      tr.dataset.groupKey = groupKey;
      const cols = [item.seq, item.time_str, item.pipe_id, item.dia, item.dist, item.defects_summary, item.note, ""];
      cols.forEach((val, idx) => {
        const td = document.createElement("td");
        td.textContent = val;
        if (idx === 5) td.classList.add("col-defects");
        tr.appendChild(td);
      });
      tr.addEventListener("dblclick", () => {
        if (openGroups.has(groupKey)) openGroups.delete(groupKey);
        else openGroups.add(groupKey);
        refreshResults();
      });
      tr.addEventListener("click", (e) => {
        if (!e.ctrlKey && !e.metaKey) selected.clear();
        const gSelKey = `group:${groupKey}`;
        if (selected.has(gSelKey)) selected.delete(gSelKey);
        else selected.add(gSelKey);
        syncSelectionClasses();
        tr.classList.toggle("row-selected", selected.has(gSelKey));
      });
      tbody.appendChild(tr);

      for (const child of item.children) {
        const childTr = buildRowTr(child, isOpen ? "" : "row-hidden");
        tbody.appendChild(childTr);
      }
    }
  }

  renderStats(data);
}

// ───────── 통계 (영상별 × 클래스별 검출 수, 오탐 제외) ─────────

function renderStats(data) {
  const head = document.getElementById("statsHead");
  const body = document.getElementById("statsBody");
  if (!head || !body) return;

  const rows = collectRows(data).filter((r) => !r.fp);
  const classes = [...new Set(rows.flatMap((r) => r.defects))].sort();
  const byVideo = {};
  for (const r of rows) {
    const v = (byVideo[r.filename] = byVideo[r.filename] || { total: 0 });
    v.total += 1;
    for (const d of r.defects) v[d] = (v[d] || 0) + 1;
  }

  head.innerHTML = "";
  const hr = document.createElement("tr");
  for (const h of ["영상", ...classes, "합계(행)"]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  head.appendChild(hr);

  body.innerHTML = "";
  const totals = { total: 0 };
  for (const [video, counts] of Object.entries(byVideo)) {
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    tdName.textContent = video;
    tr.appendChild(tdName);
    for (const c of classes) {
      const td = document.createElement("td");
      td.textContent = counts[c] || 0;
      tr.appendChild(td);
      totals[c] = (totals[c] || 0) + (counts[c] || 0);
    }
    const tdTotal = document.createElement("td");
    tdTotal.textContent = counts.total;
    tr.appendChild(tdTotal);
    totals.total += counts.total;
    body.appendChild(tr);
  }
  if (Object.keys(byVideo).length > 1) {
    const tr = document.createElement("tr");
    tr.className = "stats-total";
    const tdName = document.createElement("td");
    tdName.textContent = "전체";
    tr.appendChild(tdName);
    for (const c of classes) {
      const td = document.createElement("td");
      td.textContent = totals[c] || 0;
      tr.appendChild(td);
    }
    const tdTotal = document.createElement("td");
    tdTotal.textContent = totals.total;
    tr.appendChild(tdTotal);
    body.appendChild(tr);
  }
}

// ───────── 키보드 ↑↓ 탐지 행 네비게이션 (그룹/폴더 행 포함) + Enter로 펼치기 ─────────

// 단일 행 + 그룹(폴더) 행을 DOM 순서 그대로, 숨김/필터된 것만 제외하고 모은다.
// 그룹이 접혀 있으면 자식은 row-hidden이라 여기 안 잡히고, 그룹 행만 하나의 이동 단위가 된다.
function visibleNavTrs() {
  return [...tbody.querySelectorAll("tr[data-time-s], tr.row-group-parent")].filter(
    (tr) => !tr.classList.contains("row-hidden") && !tr.classList.contains("row-filtered")
  );
}

function navKeyOf(tr) {
  return tr.classList.contains("row-group-parent")
    ? `group:${tr.dataset.groupKey}`
    : rowKey(tr.dataset.video, Number(tr.dataset.timeS));
}

function selectNavTr(tr) {
  selected.clear();
  selected.add(navKeyOf(tr));
  syncSelectionClasses();
  tr.scrollIntoView({ block: "nearest" });

  if (tr.classList.contains("row-group-parent")) {
    // 그룹 자체는 재생 위치가 없으므로, 대표로 첫 자식 시각의 프레임을 보여준다
    const group = lastData.display.find(
      (i) => i.type === "group" && `${i.filename}|${i.dist}` === tr.dataset.groupKey
    );
    if (group && group.children[0]) activateRow(group.children[0]);
  } else {
    const video = tr.dataset.video;
    const timeS = Number(tr.dataset.timeS);
    const row = collectRows(lastData).find((r) => r.filename === video && r.time_s === timeS);
    if (row) activateRow(row);
  }
}

document.addEventListener("keydown", (e) => {
  const active = document.activeElement;
  if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
  if (!lastData) return;

  if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    const trs = visibleNavTrs();
    if (!trs.length) return;
    e.preventDefault();

    const idx = trs.findIndex((tr) => selected.has(navKeyOf(tr)));
    const nextIdx = e.key === "ArrowDown" ? Math.min(trs.length - 1, idx + 1) : Math.max(0, idx - 1);
    selectNavTr(trs[nextIdx]);
    return;
  }

  if (e.key === "Enter") {
    const trs = visibleNavTrs();
    const tr = trs.find((t) => selected.has(navKeyOf(t)));
    if (!tr || !tr.classList.contains("row-group-parent")) return;
    e.preventDefault();

    const groupKey = tr.dataset.groupKey;
    if (openGroups.has(groupKey)) openGroups.delete(groupKey);
    else openGroups.add(groupKey);
    renderResults(lastData);

    const reopened = tbody.querySelector(`tr.row-group-parent[data-group-key="${CSS.escape(groupKey)}"]`);
    if (reopened) selectNavTr(reopened);
  }
});

// 재생 중 결함 시점을 지나가면 결과 리스트에서 해당 행을 선택+스크롤 (필요하면 그룹도 자동으로 펼침)
onDetectionPass((video, timeS) => {
  if (!lastData) return;
  const row = collectRows(lastData).find((r) => r.filename === video && r.time_s === timeS);
  if (!row) return;

  const dist = (row.dist || "").trim();
  const groupKey = dist ? `${video}|${dist}` : null;
  const isGrouped = groupKey && lastData.display.some(
    (i) => i.type === "group" && `${i.filename}|${i.dist}` === groupKey
  );
  if (isGrouped && !openGroups.has(groupKey)) {
    openGroups.add(groupKey);
    renderResults(lastData);
  }

  selected.clear();
  selected.add(rowKey(video, timeS));
  syncSelectionClasses();
  const tr = tbody.querySelector(`tr[data-video="${CSS.escape(video)}"][data-time-s="${timeS}"]`);
  if (tr) tr.scrollIntoView({ block: "nearest" });
});

export async function refreshResults() {
  const data = await api.getResults();
  renderResults(data);
  const statusEl = document.getElementById("status");
  statusEl.textContent = data.analyzing ? "AI 분석중..." : "분석대기";
  const siteInput = document.getElementById("siteName");
  if (document.activeElement !== siteInput) siteInput.value = data.site_name || "";
  return data;
}

export async function deleteSelected() {
  const groupDeletes = [];
  const rowDeletes = [];
  for (const key of selected) {
    if (key.startsWith("group:")) {
      const [video, dist] = key.replace("group:", "").split("|");
      groupDeletes.push(api.deleteGroup(video, dist));
    } else {
      const [video, time_s] = key.split("|");
      rowDeletes.push(api.deleteRow(video, Number(time_s)));
    }
  }
  await Promise.all([...groupDeletes, ...rowDeletes]);
  selected.clear();
  await refreshResults();
}
