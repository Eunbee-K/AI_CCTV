const BASE = "";

async function req(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(BASE + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch (_) {}
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

export const api = {
  getQueue: () => req("GET", "/api/queue"),
  addVideos: (paths) => req("POST", "/api/queue/add", { paths }),
  clearQueue: () => req("POST", "/api/queue/clear"),
  selectVideo: (name) => req("POST", "/api/queue/select", { name }),

  runAnalysis: () => req("POST", "/api/analysis/run"),

  getResults: () => req("GET", "/api/results"),
  setSiteName: (value) => req("POST", "/api/results/site_name", { value }),
  editRow: (video, time_s, field, value) =>
    req("PATCH", "/api/results/row", { video, time_s, field, value }),
  addManualRow: (video, time_s) =>
    req("POST", "/api/results/manual_row", { video, time_s }),
  deleteRow: (video, time_s) => req("DELETE", "/api/results/row", { video, time_s }),
  deleteGroup: (video, dist) => req("DELETE", "/api/results/group", { video, dist }),

  exportExcel: (path) => req("POST", "/api/export/excel", { path }),

  getRemoteUrl: () => req("GET", "/api/config/remote_yolo_url"),
  setRemoteUrl: (url) => req("POST", "/api/config/remote_yolo_url", { url }),

  authMe: () => req("GET", "/api/auth/me"),
  logout: () => req("POST", "/api/auth/logout"),

  previewFrameUrl: (name, t) =>
    `/api/preview/frame?name=${encodeURIComponent(name)}&t=${t}`,
  previewStreamUrl: (name, startT, speed = 1) =>
    `/api/preview/stream?name=${encodeURIComponent(name)}&start_t=${startT}&speed=${speed}`,
  detectionFrameUrl: (name, timeS) =>
    `/api/results/frame?video=${encodeURIComponent(name)}&time_s=${timeS}`,
};
