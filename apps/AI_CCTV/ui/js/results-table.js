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

// 0:순번 1:시간 2:직경 3:거리 4:결함항목 5:비고
// 관로번호는 영상마다 하나뿐이라 행마다 반복할 이유가 없어 상단 입력칸으로 옮겼다.
const COL_DEFECTS = 4;
const COL_COUNT = 6;                          // 구분선 colSpan용
const READONLY_COLS = new Set([0, 1]);

function editableFieldForColumn(colIdx) {
  return ["", "", "dia", "dist", "defects", "note"][colIdx] || "";
}

/** 표시용 결함 문구: "BK(파손), DS(토사퇴적)". 엑셀 보고서와 같은 순서로 맞춘다.
 *  한글명을 모르는 코드는 코드만 그대로 쓴다. */
function defectsText(row) {
  const codes = row.defects || [];
  const kos = row.defects_ko || [];
  return codes
    .map((code, i) => {
      const ko = kos[i];
      return ko && ko !== code ? `${code}(${ko})` : String(code);
    })
    .join(", ");
}

/** 편집할 때 셀에 넣는 값. 표시는 "토사퇴적(DS)"이지만 편집은 코드("DS")로 한다 —
 *  표시 문구를 그대로 고치게 두면 "토사퇴적(DS)" 전체가 결함 코드로 저장돼버린다. */
function cellValue(row, colIdx) {
  switch (colIdx) {
    case 0: return row.seq;
    case 1: return row.time_str;
    case 2: return row.dia;
    case 3: return row.dist;
    case 4: return (row.defects || []).join(", ");
    case 5: return row.note;
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
    if (READONLY_COLS.has(colIdx)) return;   // 순번/시간/결함명/오탐은 편집 불가
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

// inGroup: 그룹(반복 결함)에 속한 자식 행이면 true → 들여쓰기로 소속을 표시한다.
// 단독 행까지 들여쓰면 무엇에 딸린 행인지 헷갈린다.
function buildRowTr(row, extraClass, inGroup) {
  const tr = document.createElement("tr");
  tr.className = `row-child ${inGroup ? "row-in-group " : ""}${extraClass || ""}`;
  tr.dataset.video = row.filename;
  tr.dataset.timeS = row.time_s;
  if (row.fp) tr.classList.add("row-fp");
  if (!rowVisible(row)) tr.classList.add("row-filtered");

  const key = rowKey(row.filename, row.time_s);
  if (selected.has(key)) tr.classList.add("row-selected");

  const cols = [row.seq, row.time_str, row.dia, row.dist, defectsText(row), row.note];
  cols.forEach((val, idx) => {
    const td = document.createElement("td");
    td.textContent = val;
    if (idx === COL_DEFECTS) td.classList.add("col-defects");
    makeCellEditable(td, row, idx);
    tr.appendChild(td);
  });

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

/** 그룹(같은 거리로 묶인 반복 결함) 펼치기/접기.
 *  서버를 다시 부르지 않고 마지막 데이터로 즉시 다시 그린다 —
 *  refreshResults()는 네트워크 왕복이라 느리고, 실패하면 토글이 조용히 먹통이 된다. */
function toggleGroup(groupKey) {
  if (openGroups.has(groupKey)) openGroups.delete(groupKey);
  else openGroups.add(groupKey);
  selected.clear();
  selected.add(`group:${groupKey}`);   // 토글 후에도 Enter로 계속 조작할 수 있게 선택 유지
  if (lastData) renderResults(lastData);
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
      td.colSpan = COL_COUNT;
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
      if (isOpen) tr.classList.add("is-open");
      // 다시 그려도 선택 표시가 유지되게 (토글 후 Enter 연속 조작에 필요)
      if (selected.has(`group:${groupKey}`)) tr.classList.add("row-selected");

      const cols = [item.seq, item.time_str, item.dia, item.dist,
                    item.defects_summary, item.note];
      cols.forEach((val, idx) => {
        const td = document.createElement("td");
        if (idx === 0) {
          // 접힘/펼침 상태를 눈으로 알 수 있게 삼각형을 붙인다. 클릭해도 토글된다.
          const caret = document.createElement("span");
          caret.className = "group-caret";
          caret.textContent = isOpen ? "▼" : "▶";
          caret.title = "클릭 / 더블클릭 / Enter 로 펼치기·접기";
          caret.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleGroup(groupKey);
          });
          td.appendChild(caret);
          td.appendChild(document.createTextNode(String(val)));
        } else {
          td.textContent = val;
        }
        if (idx === COL_DEFECTS) td.classList.add("col-defects");
        tr.appendChild(td);
      });

      tr.addEventListener("dblclick", (e) => {
        e.preventDefault();
        toggleGroup(groupKey);
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
        const childTr = buildRowTr(child, isOpen ? "" : "row-hidden", true);
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
    toggleGroup(groupKey);   // 더블클릭과 완전히 같은 동작

    const reopened = tbody.querySelector(`tr.row-group-parent[data-group-key="${CSS.escape(groupKey)}"]`);
    if (reopened) reopened.scrollIntoView({ block: "nearest" });
    return;
  }

  // Del: 선택한 결함 행(또는 그룹)을 삭제 — 잘못 잡힌 결함을 빠르게 지운다
  if (e.key === "Delete") {
    if (!selected.size) return;
    e.preventDefault();
    deleteSelected();
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

  // 관로번호는 영상별 값이라 현재 선택된 영상의 것을 상단 입력칸에 보여준다.
  // (입력 중일 때는 덮어쓰지 않는다)
  const pipeInput = document.getElementById("pipeId");
  if (pipeInput && document.activeElement !== pipeInput) {
    const cur = (data.videos || []).find((v) => v.name === getCurrentVideo());
    pipeInput.value = cur ? cur.pipe_id || "" : "";
    pipeInput.disabled = !cur;
  }

  // 관로 구분(신설/노후) 체크 상태를 서버 값과 맞춘다
  const condNew = document.getElementById("condNew");
  const condOld = document.getElementById("condOld");
  if (condNew && condOld) {
    condNew.checked = data.pipe_condition === "신설";
    condOld.checked = data.pipe_condition === "노후";
  }
  return data;
}

/** 현재 선택된 행(그룹 포함) 개수. [- 행 삭제] 버튼이 안내를 띄울지 판단하는 데 쓴다. */
export function getSelectedCount() {
  return selected.size;
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
