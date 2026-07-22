import { api } from "./api.js";
import { filterBoxes, onFilterChange } from "./filters.js";

const previewImg = document.getElementById("previewImg");
const previewViewport = document.getElementById("previewViewport");
const videoCard = document.getElementById("videoCard");
const boxOverlay = document.getElementById("boxOverlay");
const btnPlay = document.getElementById("btnPlay");
const speedGroup = document.getElementById("speedGroup");
const slider = document.getElementById("seekSlider");
const lblTime = document.getElementById("lblTime");
const videoNameEl = document.getElementById("videoName");
const videoListEl = document.getElementById("videoList");

const player = {
  name: null,
  duration_s: 0,
  currentTime: 0,
  playing: false,
  speed: 1,
  _tick: null,
};

// video -> { time_s -> boxes[] } (정규화 좌표, results-table이 채워줌)
let detectionMap = {};
// 현재 오버레이에 고정 표시 중인 박스 (행 클릭 시). null이면 재생 시간 기준 자동.
let pinnedBoxes = null;
// 재생 중 결함 시점을 지날 때 결과 리스트를 하이라이트하기 위한 콜백
let detectionPassCb = null;
let lastPassKey = null;

export function onDetectionPass(cb) {
  detectionPassCb = cb;
}

// 재생 중 현재 초에 결함 탐지가 있으면(해당 초를 막 지나가는 시점) 콜백 1회 호출.
// 같은 초에서 매 tick(250ms)마다 중복 호출되지 않도록 lastPassKey로 방지.
function checkDetectionPass() {
  if (!player.name || !detectionPassCb) return;
  const sec = Math.floor(player.currentTime);
  const byTime = detectionMap[player.name];
  if (!byTime || !byTime[sec] || !byTime[sec].length) return;
  const passKey = `${player.name}|${sec}`;
  if (passKey === lastPassKey) return;
  lastPassKey = passKey;
  detectionPassCb(player.name, sec);
}

function fmt(sec) {
  sec = Math.max(0, Math.floor(sec));
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function showSnapshot(t) {
  if (!player.name) return;
  previewImg.src = api.previewFrameUrl(player.name, t);
}

function updateSliderUI() {
  slider.max = player.duration_s || 100;
  slider.value = player.currentTime;
  lblTime.textContent = fmt(player.currentTime);
}

// ───────── 박스 오버레이 ─────────

export function setDetections(map) {
  detectionMap = map || {};
  renderOverlay();
}

function boxesAt(name, t) {
  const byTime = detectionMap[name];
  if (!byTime) return [];
  // 프레임은 2초 간격으로 추출되므로 t 또는 t-1 초의 탐지를 현재 화면에 매칭
  const sec = Math.floor(t);
  return byTime[sec] || byTime[sec - 1] || [];
}

// 오버레이 div를 (레터박스 여백을 뺀) 실제 이미지 표시 영역에 정확히 맞춘다
function syncOverlayRect() {
  boxOverlay.style.left = `${previewImg.offsetLeft}px`;
  boxOverlay.style.top = `${previewImg.offsetTop}px`;
  boxOverlay.style.width = `${previewImg.offsetWidth}px`;
  boxOverlay.style.height = `${previewImg.offsetHeight}px`;
}

previewImg.addEventListener("load", () => {
  syncOverlayRect();
  renderOverlay();
});
window.addEventListener("resize", syncOverlayRect);

function renderOverlay() {
  syncOverlayRect();
  const boxes = pinnedBoxes !== null ? pinnedBoxes : boxesAt(player.name, player.currentTime);
  const visible = filterBoxes(boxes);
  boxOverlay.innerHTML = "";
  for (const b of visible) {
    const [x1, y1, x2, y2] = b.nxyxy || [];
    if (x2 == null) continue;
    const div = document.createElement("div");
    div.className = "det-box";
    div.style.left = `${x1 * 100}%`;
    div.style.top = `${y1 * 100}%`;
    div.style.width = `${(x2 - x1) * 100}%`;
    div.style.height = `${(y2 - y1) * 100}%`;
    const lbl = document.createElement("span");
    lbl.className = "det-label";
    lbl.textContent =
      typeof b.confidence === "number"
        ? `${b.label} ${(b.confidence * 100).toFixed(0)}%`
        : b.label;
    div.appendChild(lbl);
    boxOverlay.appendChild(div);
  }
}

onFilterChange(renderOverlay);

// ───────── 기본 플레이어 ─────────

export function getCurrentVideo() {
  return player.name;
}

export function getCurrentTime() {
  return Math.floor(player.currentTime);
}

export async function selectVideo(name) {
  pause();
  const meta = await api.selectVideo(name);
  player.name = name;
  player.duration_s = meta.duration_s;
  player.currentTime = 0;
  pinnedBoxes = null;
  videoNameEl.textContent = name;
  updateSliderUI();
  showSnapshot(0);
  renderOverlay();
  resetZoom();
  [...videoListEl.children].forEach((li) =>
    li.classList.toggle("selected", li.dataset.name === name)
  );
}

export function seek(t) {
  pinnedBoxes = null;
  lastPassKey = null; // 시간을 옮기면 같은 초를 다시 지나가도 하이라이트되게 초기화
  player.currentTime = Math.min(Math.max(0, t), player.duration_s || t);
  updateSliderUI();
  if (player.playing) {
    startStream();
  } else {
    showSnapshot(player.currentTime);
    renderOverlay();
  }
}

// 결함 행 클릭 시: 해당 시각의 프레임 + 그 행의 박스를 고정 표시.
// (재생하거나 슬라이더를 움직이면 다시 시간 기준 자동 오버레이로 복귀)
export function showDetectionFrame(name, timeS, boxes) {
  player.playing = false;
  btnPlay.textContent = "▶";
  clearInterval(player._tick);
  player.currentTime = Math.min(Math.max(0, timeS), player.duration_s || timeS);
  updateSliderUI();
  showSnapshot(timeS);
  pinnedBoxes = boxes || boxesAt(name, timeS);
  renderOverlay();
}

function startStream() {
  if (!player.name) return;
  pinnedBoxes = null;
  previewImg.src = api.previewStreamUrl(player.name, player.currentTime, player.speed);
  clearInterval(player._tick);
  const startedAt = Date.now();
  const baseTime = player.currentTime;
  const speed = player.speed;
  player._tick = setInterval(() => {
    const elapsed = ((Date.now() - startedAt) / 1000) * speed;
    player.currentTime = baseTime + elapsed;
    if (player.duration_s && player.currentTime >= player.duration_s) {
      player.currentTime = player.duration_s;
      pause();
      return;
    }
    updateSliderUI();
    renderOverlay(); // 재생 중 실시간 박스 표시
    checkDetectionPass();
  }, 250);
}

export function play() {
  if (!player.name || player.playing) return;
  player.playing = true;
  btnPlay.textContent = "❚❚";
  startStream();
}

export function pause() {
  player.playing = false;
  btnPlay.textContent = "▶";
  clearInterval(player._tick);
  if (player.name) {
    showSnapshot(player.currentTime);
    renderOverlay();
  }
}

export function togglePlay() {
  if (player.playing) pause();
  else play();
}

// ───────── 배속 ─────────

function setSpeed(s) {
  player.speed = s;
  [...speedGroup.querySelectorAll(".speed-btn")].forEach((b) =>
    b.classList.toggle("active", Number(b.dataset.speed) === s)
  );
  if (player.playing) startStream(); // 새 배속으로 스트림 재시작
}

speedGroup.addEventListener("click", (e) => {
  const btn = e.target.closest(".speed-btn");
  if (btn) setSpeed(Number(btn.dataset.speed));
});

// ───────── 줌/팬 ─────────

const zoom = { scale: 1, tx: 0, ty: 0 };

function applyZoom() {
  previewViewport.style.transform = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`;
}

function resetZoom() {
  zoom.scale = 1;
  zoom.tx = 0;
  zoom.ty = 0;
  applyZoom();
}

videoCard.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = videoCard.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const newScale = Math.min(8, Math.max(1, zoom.scale * factor));
  // 마우스 위치를 기준점으로 확대/축소
  zoom.tx = mx - ((mx - zoom.tx) / zoom.scale) * newScale;
  zoom.ty = my - ((my - zoom.ty) / zoom.scale) * newScale;
  zoom.scale = newScale;
  if (zoom.scale === 1) {
    zoom.tx = 0;
    zoom.ty = 0;
  }
  applyZoom();
}, { passive: false });

let dragging = null;
videoCard.addEventListener("mousedown", (e) => {
  if (zoom.scale <= 1) return;
  dragging = { x: e.clientX - zoom.tx, y: e.clientY - zoom.ty };
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  zoom.tx = e.clientX - dragging.x;
  zoom.ty = e.clientY - dragging.y;
  applyZoom();
});
window.addEventListener("mouseup", () => {
  dragging = null;
});
videoCard.addEventListener("dblclick", resetZoom);

// ───────── 영상 목록 ─────────

export function renderQueue(list) {
  videoListEl.innerHTML = "";
  for (const item of list) {
    const li = document.createElement("li");
    li.textContent = item.name;
    li.dataset.name = item.name;
    if (item.name === player.name) li.classList.add("selected");
    li.addEventListener("click", () => selectVideo(item.name));
    videoListEl.appendChild(li);
  }
}

btnPlay.addEventListener("click", togglePlay);
slider.addEventListener("input", (e) => {
  lblTime.textContent = fmt(Number(e.target.value));
});
slider.addEventListener("change", (e) => {
  seek(Number(e.target.value));
});

// ───────── 키보드: 스페이스=재생/정지, 좌우=1초 이동 ─────────
document.addEventListener("keydown", (e) => {
  const active = document.activeElement;
  if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
  if (!player.name) return;

  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    seek(player.currentTime - 1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    seek(player.currentTime + 1);
  }
});
