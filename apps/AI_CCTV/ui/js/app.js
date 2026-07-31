import { api } from "./api.js";
import { connectWs, on } from "./ws.js";
import { renderQueue, getCurrentVideo, getCurrentTime, selectVideo } from "./player.js";
import { refreshResults, deleteSelected, getSelectedCount } from "./results-table.js";

const logConsole = document.getElementById("logConsole");
const statusEl = document.getElementById("status");
const siteNameEl = document.getElementById("siteName");

/** 선택된 관로 구분("신설"/"노후"). 아무것도 안 골랐으면 "". */
function getPipeCondition() {
  if (document.getElementById("condNew").checked) return "신설";
  if (document.getElementById("condOld").checked) return "노후";
  return "";
}

function appendLog(msg) {
  const ts = new Date().toLocaleTimeString("ko-KR", { hour12: false });
  const line = document.createElement("div");
  line.innerHTML = `<span class="ts">[${ts}]</span> <span class="lvl-${msg.level}">${escapeHtml(msg.msg)}</span>`;
  logConsole.appendChild(line);
  logConsole.scrollTop = logConsole.scrollHeight;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function refreshQueue() {
  const list = await api.getQueue();
  renderQueue(list);
  // 목록이 비어있을 때만 "끌어놓으세요" 안내를 보여준다
  const hint = document.getElementById("dropHint");
  if (hint) hint.hidden = list.length > 0;
  if (!getCurrentVideo() && list.length) {
    // 영상 하나가 안 열려도(코덱/삭제됨) 여기서 죽으면 결과표까지 안 그려진다.
    // 로그만 남기고 나머지 초기화는 계속 진행한다.
    try {
      await selectVideo(list[0].name);
    } catch (e) {
      appendLog({ level: "ERROR", msg: `영상을 열 수 없습니다 (${list[0].name}): ${e.message}` });
    }
  }
}

function waitForPywebview() {
  return new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve(true);
    let settled = false;
    window.addEventListener("pywebviewready", () => {
      if (!settled) {
        settled = true;
        resolve(true);
      }
    });
    // pywebview 없이 브라우저에서 개발 중 테스트할 때를 위한 폴백
    setTimeout(() => {
      if (!settled) {
        settled = true;
        resolve(false);
      }
    }, 1500);
  });
}

// ───────── 좌우 비율 조절 (스플리터 드래그) ─────────

const SPLIT_KEY = "aicctv.splitRatio";
const SPLIT_MIN = 0.2, SPLIT_MAX = 0.8;   // 한쪽이 완전히 사라지지 않게 제한

function applySplit(ratio) {
  const app = document.querySelector(".app");
  const pct = (Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, ratio)) * 100).toFixed(2);
  // 스플리터 폭(6px)은 고정, 나머지를 좌:우 비율로 나눈다
  app.style.gridTemplateColumns = `${pct}fr 6px ${(100 - pct).toFixed(2)}fr`;
}

function setupSplitter() {
  const splitter = document.getElementById("splitter");
  const app = document.querySelector(".app");
  if (!splitter || !app) return;

  const saved = Number(localStorage.getItem(SPLIT_KEY));
  applySplit(saved > 0 ? saved : 0.5);   // 기본 1:1

  let dragging = false;
  const onMove = (e) => {
    if (!dragging) return;
    const rect = app.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    applySplit(ratio);
  };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("splitting");
    const cols = app.style.gridTemplateColumns.match(/([\d.]+)fr/);
    if (cols) localStorage.setItem(SPLIT_KEY, String(Number(cols[1]) / 100));
  };

  splitter.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    // 드래그 중 영상/표 위에서 텍스트가 선택되거나 커서가 바뀌지 않게
    document.body.classList.add("splitting");
  });
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  splitter.addEventListener("dblclick", () => {
    applySplit(0.5);
    localStorage.setItem(SPLIT_KEY, "0.5");
  });
}

async function init() {
  const hasPywebview = await waitForPywebview();
  window.__hasPywebview = hasPywebview;

  connectWs();
  on("log", appendLog);
  on("progress", (msg) => {
    statusEl.textContent = `AI 분석중 (${msg.index}/${msg.total})`;
  });
  on("result_update", refreshResults);
  on("batch_done", async (msg) => {
    await refreshResults();
    const totalRows = msg.stats && typeof msg.stats.yolo === "number" ? msg.stats.yolo : 0;
    if (msg.errors && msg.errors.length && totalRows === 0) {
      alert(`결과가 없습니다.\n${[...new Set(msg.errors)].slice(0, 3).join("\n")}`);
    } else {
      alert(`분석 완료!\n[YOLO] 감지 프레임: ${totalRows}`);
    }
  });

  // ── 영상 추가: 파일 선택 + 드래그앤드롭 업로드 ──
  const fileInput = document.getElementById("videoFileInput");
  const queueCard = document.getElementById("queueCard");
  const uploadBar = document.getElementById("uploadBar");
  const uploadBarFill = document.getElementById("uploadBarFill");
  const uploadBarText = document.getElementById("uploadBarText");

  const fmtMB = (bytes) => (bytes / 1024 / 1024).toFixed(1) + "MB";

  async function uploadFiles(fileList) {
    const files = [...fileList];
    if (!files.length) return;

    uploadBar.hidden = false;
    uploadBarFill.style.width = "0%";
    uploadBarText.textContent = `업로드 준비 중… (${files.length}개)`;
    appendLog({ level: "INFO", msg: `영상 업로드 시작: ${files.map((f) => f.name).join(", ")}` });

    try {
      const res = await api.uploadVideos(files, (loaded, total) => {
        const pct = total ? Math.round((loaded / total) * 100) : 0;
        uploadBarFill.style.width = pct + "%";
        uploadBarText.textContent = `업로드 중 ${pct}% (${fmtMB(loaded)} / ${fmtMB(total)})`;
      });
      uploadBarText.textContent = "서버에서 파일 확인 중…";

      if (res.saved && res.saved.length) {
        appendLog({ level: "INFO", msg: `업로드 완료: ${res.saved.join(", ")}` });
      }
      if (res.skipped && res.skipped.length) {
        appendLog({ level: "ERROR", msg: `제외된 파일: ${res.skipped.join(", ")}` });
        alert(`일부 파일을 추가하지 못했습니다:\n\n${res.skipped.join("\n")}`);
      }
      await refreshQueue();
    } catch (e) {
      appendLog({ level: "ERROR", msg: `업로드 실패: ${e.message}` });
      alert(`업로드 실패: ${e.message}`);
    } finally {
      uploadBar.hidden = true;
    }
  }

  document.getElementById("btnAddVideos").addEventListener("click", async () => {
    if (window.__hasPywebview) {
      // 데스크톱(exe): 네이티브 파일창으로 로컬 경로를 그대로 큐에 넣는다(복사 불필요)
      await window.pywebview.api.open_video_dialog();
      await refreshQueue();
    } else {
      fileInput.click();   // 웹: 브라우저 파일 선택 → 업로드
    }
  });

  fileInput.addEventListener("change", async () => {
    await uploadFiles(fileInput.files);
    fileInput.value = "";   // 같은 파일을 다시 골라도 change가 뜨도록 초기화
  });

  // 드래그앤드롭 — 목록 카드 위에 끌어놓으면 업로드
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach((ev) =>
    queueCard.addEventListener(ev, (e) => {
      stop(e);
      e.dataTransfer.dropEffect = "copy";
      queueCard.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    queueCard.addEventListener(ev, (e) => {
      stop(e);
      // dragleave가 자식 요소로 옮겨갈 때도 뜨므로, 카드 밖으로 나간 경우만 해제
      if (ev === "dragleave" && queueCard.contains(e.relatedTarget)) return;
      queueCard.classList.remove("drag-over");
    })
  );
  queueCard.addEventListener("drop", async (e) => {
    if (window.__hasPywebview) {
      alert("데스크톱 버전에서는 [파일추가] 버튼을 사용하세요.");
      return;
    }
    const dropped = [...(e.dataTransfer.files || [])];
    if (!dropped.length) return;
    await uploadFiles(dropped);
  });

  // 브라우저 창 아무데나 떨어뜨렸을 때 영상이 재생되며 페이지가 날아가는 것 방지
  ["dragover", "drop"].forEach((ev) =>
    window.addEventListener(ev, (e) => {
      if (!queueCard.contains(e.target)) e.preventDefault();
    })
  );

  document.getElementById("btnClearVideos").addEventListener("click", async () => {
    await api.clearQueue();
    await refreshQueue();
    await refreshResults();
  });

  document.getElementById("btnRun").addEventListener("click", async () => {
    // 관로 구분을 안 고르면 결함 판정 기준이 정해지지 않아 분석 의미가 없다.
    if (!getPipeCondition()) {
      alert("관로 구분을 선택하세요.\n\n[신설] 또는 [노후] 중 하나를 체크한 뒤 분석을 실행할 수 있습니다.");
      document.getElementById("condNew").focus();
      return;
    }
    try {
      await api.runAnalysis();
    } catch (e) {
      alert(e.message);
    }
  });

  document.getElementById("btnAddRow").addEventListener("click", async () => {
    const video = getCurrentVideo();
    if (!video) {
      alert("먼저 영상을 선택하세요.\n(왼쪽 '분석 대기 영상' 목록에서 클릭)");
      return;
    }
    try {
      await api.addManualRow(video, getCurrentTime());
      await refreshResults();
      appendLog({ level: "INFO", msg: `행 추가: ${video} ${getCurrentTime()}초` });
    } catch (e) {
      alert(`행 추가 실패: ${e.message}`);
    }
  });

  document.getElementById("btnDelRow").addEventListener("click", async () => {
    if (!getSelectedCount()) {
      alert("삭제할 행을 먼저 선택하세요.\n(행을 클릭해서 선택 · Ctrl+클릭으로 여러 개 선택)");
      return;
    }
    try {
      await deleteSelected();
    } catch (e) {
      alert(`행 삭제 실패: ${e.message}`);
    }
  });

  document.getElementById("btnExport").addEventListener("click", async () => {
    const btn = document.getElementById("btnExport");
    btn.disabled = true;
    try {
      if (window.__hasPywebview) {
        // 데스크톱(exe): 네이티브 저장 대화상자로 로컬 경로를 직접 고른다
        const path = await window.pywebview.api.save_excel_dialog();
        if (!path) return;
        await api.exportExcel(path);
        appendLog({ level: "INFO", msg: `보고서 저장: ${path}` });
        alert(`저장되었습니다.\n\n${path}`);
      } else {
        // 웹: 브라우저가 내려받는다(저장 위치는 브라우저 설정/저장 대화상자가 결정).
        // 서버 경로에 저장하면 접속한 사람 PC가 아니라 서버에 파일이 생긴다.
        const name = await api.downloadExcel();
        appendLog({ level: "INFO", msg: `보고서 다운로드: ${name}` });
      }
    } catch (e) {
      appendLog({ level: "ERROR", msg: `보고서 출력 실패: ${e.message}` });
      alert(`보고서 출력 실패: ${e.message}`);
    } finally {
      btn.disabled = false;
    }
  });

  siteNameEl.addEventListener("change", () => {
    api.setSiteName(siteNameEl.value);
  });

  // 관로 구분(신설/노후) — 체크박스지만 둘 중 하나만 켜지게 한다(라디오처럼).
  const condNew = document.getElementById("condNew");
  const condOld = document.getElementById("condOld");
  for (const [me, other] of [[condNew, condOld], [condOld, condNew]]) {
    me.addEventListener("change", async () => {
      if (me.checked) other.checked = false;   // 하나를 켜면 다른 쪽은 꺼진다
      const value = getPipeCondition();
      try {
        await api.setPipeCondition(value);
        appendLog({ level: "INFO", msg: value ? `관로 구분: ${value}` : "관로 구분 선택 해제" });
      } catch (e) {
        appendLog({ level: "ERROR", msg: `관로 구분 저장 실패: ${e.message}` });
      }
    });
  }

  // 관로번호 — 표에서 열을 빼고 이리로 옮겼다. 현재 선택된 영상의 값을 고친다.
  const pipeIdEl = document.getElementById("pipeId");
  pipeIdEl.addEventListener("change", async () => {
    const video = getCurrentVideo();
    if (!video) return;
    // pipe_id는 영상 단위 값이라 time_s는 쓰이지 않는다(0을 넘겨도 무방)
    await api.editRow(video, 0, "pipe_id", pipeIdEl.value);
    await refreshResults();
  });

  // 로그인 사용자 표시 + 로그아웃 (웹 배포에서 AUTH_ENABLED=1일 때만 노출)
  try {
    const meRes = await api.authMe();
    if (meRes.auth_enabled && meRes.user) {
      document.getElementById("userBar").hidden = false;
      document.getElementById("userBadge").textContent = "👤 " + meRes.user;
    }
  } catch (_) {}
  document.getElementById("btnLogout").addEventListener("click", async () => {
    try { await api.logout(); } catch (_) {}
    window.location.href = "/login";
  });

  // Colab 추론 서버 연결 (링크는 index.html의 href="" 에 직접 넣어 사용)
  const remoteInput = document.getElementById("remoteUrl");
  const colabStatus = document.getElementById("colabStatus");
  const btnConnect = document.getElementById("btnConnectRemote");
  // reachable이 undefined면(=저장된 주소를 불러온 경우) 아직 확인 전 상태로 둔다
  const setColabStatus = (url, reachable) => {
    if (!url) {
      colabStatus.textContent = "미연결";
      colabStatus.className = "status";
      return;
    }
    if (reachable === undefined) {
      colabStatus.textContent = "주소 있음 (미확인)";
      colabStatus.className = "status";
      return;
    }
    colabStatus.textContent = reachable ? "연결됨" : "연결 실패";
    colabStatus.className = reachable ? "status ok" : "status err";
  };

  try {
    const r = await api.getRemoteUrl();
    remoteInput.value = r.url || "";
    setColabStatus(r.url, undefined);
  } catch (_) {}

  btnConnect.addEventListener("click", async () => {
    const raw = remoteInput.value.trim();
    btnConnect.disabled = true;
    colabStatus.textContent = "확인 중…";
    colabStatus.className = "status";
    try {
      const r = await api.setRemoteUrl(raw);
      remoteInput.value = r.url || "";
      setColabStatus(r.url, r.url ? r.reachable : undefined);
      if (!r.url) {
        appendLog({ level: "INFO", msg: "추론 서버 연결 해제 (로컬 추론)" });
      } else if (r.reachable) {
        appendLog({ level: "INFO", msg: `추론 서버 연결됨: ${r.url} — ${r.detail}` });
      } else {
        appendLog({ level: "ERROR", msg: `추론 서버 연결 실패: ${r.detail}` });
        alert(`추론 서버에 연결하지 못했습니다.\n\n${r.detail}`);
      }
    } catch (e) {
      setColabStatus(raw, false);
      appendLog({ level: "ERROR", msg: `연결 요청 실패: ${e.message}` });
    } finally {
      btnConnect.disabled = false;
    }
  });

  setupSplitter();

  // 탭 전환 (결과표 / 통계)
  const tabBtns = [...document.querySelectorAll(".tab-btn")];
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-page").forEach((page) => {
        page.hidden = page.id !== btn.dataset.tab;
      });
    });
  });

  await refreshQueue();
  await refreshResults();
}

init();
