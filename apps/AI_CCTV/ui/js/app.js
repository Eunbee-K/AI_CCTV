import { api } from "./api.js";
import { connectWs, on } from "./ws.js";
import { renderQueue, getCurrentVideo, getCurrentTime, selectVideo } from "./player.js";
import { refreshResults, deleteSelected } from "./results-table.js";

const logConsole = document.getElementById("logConsole");
const statusEl = document.getElementById("status");
const siteNameEl = document.getElementById("siteName");

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
  if (!getCurrentVideo() && list.length) {
    await selectVideo(list[0].name);
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

  document.getElementById("btnAddVideos").addEventListener("click", async () => {
    if (window.__hasPywebview) {
      await window.pywebview.api.open_video_dialog();
    } else {
      const input = prompt("영상 파일의 전체 경로를 입력하세요 (여러개는 ; 로 구분):");
      if (input) {
        await api.addVideos(input.split(";").map((s) => s.trim()).filter(Boolean));
      }
    }
    await refreshQueue();
  });

  document.getElementById("btnClearVideos").addEventListener("click", async () => {
    await api.clearQueue();
    await refreshQueue();
    await refreshResults();
  });

  document.getElementById("btnRun").addEventListener("click", async () => {
    try {
      await api.runAnalysis();
    } catch (e) {
      alert(e.message);
    }
  });

  document.getElementById("btnAddRow").addEventListener("click", async () => {
    const video = getCurrentVideo();
    if (!video) return;
    await api.addManualRow(video, getCurrentTime());
    await refreshResults();
  });

  document.getElementById("btnDelRow").addEventListener("click", async () => {
    await deleteSelected();
  });

  document.getElementById("btnExport").addEventListener("click", async () => {
    let path = null;
    if (window.__hasPywebview) {
      path = await window.pywebview.api.save_excel_dialog();
    } else {
      path = prompt("저장할 xlsx 전체 경로를 입력하세요:", "report.xlsx");
    }
    if (!path) return;
    try {
      await api.exportExcel(path);
      alert("저장되었습니다.");
    } catch (e) {
      alert(`저장 실패: ${e.message}`);
    }
  });

  siteNameEl.addEventListener("change", () => {
    api.setSiteName(siteNameEl.value);
  });

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
