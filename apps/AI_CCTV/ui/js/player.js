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

// 영상이 바뀌면 결과표를 그 영상 것으로 다시 불러와야 한다 (results-table이 구독).
let videoChangeCb = null;
export function onVideoChange(cb) {
  videoChangeCb = cb;
}

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

const LABEL_H_PX = 16;   // .det-label의 대략적인 높이(폰트 11px + 패딩)

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
    // 라벨은 기본으로 박스 위에 붙는데, 박스가 화면 위쪽에 닿아 있으면(큰 박스가
    // 대표적) 라벨이 영상 밖으로 잘려서 신뢰도를 못 읽는다. 그때는 박스 안쪽에 넣는다.
    const labelRoomPx = y1 * boxOverlay.offsetHeight;
    if (labelRoomPx < LABEL_H_PX) lbl.classList.add("inside");
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
  if (videoChangeCb) await videoChangeCb(name);
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

// 지금 열려 있는 스트림의 식별자. 정지할 때 서버에 "이 스트림 어디까지 갔냐"고
// 물어보기 위해 필요하다.
let streamSid = "";

function startStream() {
  if (!player.name) return;
  pinnedBoxes = null;
  streamSid = String(Date.now());
  previewImg.src = api.previewStreamUrl(
    player.name, player.currentTime, player.speed, streamSid);
  clearInterval(player._tick);
  // 벽시계 기준점. 서버가 알려준 실제 위치로 수시로 다시 잡는다(아래 참고).
  let startedAt = Date.now();
  let baseTime = player.currentTime;
  const speed = player.speed;
  const sid = streamSid;
  let syncing = false;
  let ticks = 0;
  player._tick = setInterval(() => {
    const elapsed = ((Date.now() - startedAt) / 1000) * speed;
    player.currentTime = baseTime + elapsed;

    // **벽시계는 스트림보다 앞서 나간다** — 프레임마다 1/fps를 자는 위에
    // 디코딩·인코딩·전송 시간이 더 붙기 때문이다. 2초에 한 번 서버에 실제
    // 위치를 물어 기준점을 다시 잡는다. (currentTime만 덮어쓰면 다음 tick이
    // 옛 기준점으로 도로 계산해버리므로 baseTime/startedAt을 같이 옮긴다.)
    if (++ticks % 8 === 0 && !syncing && sid) {
      syncing = true;
      api.streamPos(player.name, sid)
        .then((r) => {
          const t = r && r.t;
          if (player.playing && sid === streamSid &&
              typeof t === "number" && Number.isFinite(t)) {
            baseTime = t;
            startedAt = Date.now();
            player.currentTime = t;
          }
        })
        .catch(() => {})
        .finally(() => { syncing = false; });
    }

    if (player.duration_s && player.currentTime >= player.duration_s) {
      player.currentTime = player.duration_s;
      // 끝까지 간 경우는 위치가 확정이다. 서버에 되묻으면 스트림이 뒤처진 만큼
      // 뒤로 끌려가므로 그냥 끝에 세운다.
      pause(true);
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

// keepTime=true면 서버에 위치를 되묻지 않고 현재 값 그대로 멈춘다
// (영상 끝처럼 위치가 이미 확정된 경우).
export function pause(keepTime = false) {
  const wasPlaying = player.playing;
  player.playing = false;
  btnPlay.textContent = "▶";
  clearInterval(player._tick);
  if (!player.name) return;

  // **화면에 실제로 떠 있던 프레임에서 멈춘다.**
  //
  // 재생 위치는 벽시계로 세는데, MJPEG 스트림은 프레임마다 1/fps를 자고 그 위에
  // 디코딩·전송 시간이 더 붙어서 늘 실시간보다 뒤처진다. 벽시계 값으로 스냅샷을
  // 받으면 정지하는 순간 화면이 앞으로 건너뛰고, 결함이 촘촘한 구간에서는 마치
  // "결함 지점으로 제멋대로 이동"하는 것처럼 보인다.
  //
  // 그래서 서버에 스트림이 실제로 보낸 마지막 프레임 시각을 물어보고 그 자리에
  // 멈춘다. 못 물어보면(스트림이 이미 끊겼거나 요청 실패) 예전처럼 벽시계 값을 쓴다.
  const name = player.name;
  const settle = (t) => {
    if (player.playing || player.name !== name) return;   // 그 사이 다시 재생했으면 건드리지 않는다
    if (typeof t === "number" && Number.isFinite(t)) {
      player.currentTime = Math.min(Math.max(0, t), player.duration_s || t);
      updateSliderUI();
    }
    showSnapshot(player.currentTime);
    renderOverlay();
  };

  if (wasPlaying && streamSid && !keepTime) {
    api.streamPos(name, streamSid)
      .then((r) => settle(r && r.t))
      .catch(() => settle(null));
  } else {
    settle(null);
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

const MIN_SCALE = 0.25;   // 축소 한계 (박스 전체를 한눈에 보려고 줄일 때)
const MAX_SCALE = 8;

const zoom = { scale: 1, tx: 0, ty: 0 };
const zoomBadge = document.getElementById("zoomBadge");

function applyZoom() {
  previewViewport.style.transform = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`;
  if (zoomBadge) {
    zoomBadge.textContent = `${Math.round(zoom.scale * 100)}%`;
    zoomBadge.hidden = Math.abs(zoom.scale - 1) < 0.01;
  }
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
  let newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, zoom.scale * factor));

  // 1.2배씩 곱/나누기라 부동소수 오차로 정확히 1.0이 안 된다. 근처면 1로 스냅.
  if (Math.abs(newScale - 1) < 0.02) newScale = 1;

  if (newScale === 1) {
    zoom.tx = 0;
    zoom.ty = 0;
  } else if (newScale < 1) {
    // 축소할 때는 마우스 기준으로 당기면 구석으로 쏠려 보기 나쁘다 → 가운데 정렬
    zoom.tx = (rect.width * (1 - newScale)) / 2;
    zoom.ty = (rect.height * (1 - newScale)) / 2;
  } else {
    // 확대는 마우스 위치를 기준점으로 (보던 지점이 유지된다)
    zoom.tx = mx - ((mx - zoom.tx) / zoom.scale) * newScale;
    zoom.ty = my - ((my - zoom.ty) / zoom.scale) * newScale;
  }
  zoom.scale = newScale;
  applyZoom();
}, { passive: false });

let dragging = null;
videoCard.addEventListener("mousedown", (e) => {
  if (zoom.scale <= 1) return;   // 축소/원본 상태에서는 전체가 보이므로 팬 불필요
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
    li.addEventListener("click", () => {
      // 직접 고른 것은 분석이 화면을 끌고 다니지 말라는 뜻이다(app.js가 듣는다).
      window.dispatchEvent(new CustomEvent("video-picked-by-user"));
      selectVideo(item.name);
    });
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
