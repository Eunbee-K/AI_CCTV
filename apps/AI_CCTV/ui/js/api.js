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

/** 영상 업로드. 파일이 크므로 진행률을 받을 수 있는 XHR을 쓴다(fetch는 업로드 진행률 미지원). */
function uploadVideos(fileList, onProgress) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of fileList) fd.append("files", f, f.name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", BASE + "/api/queue/upload");
    xhr.upload.addEventListener("progress", (e) => {
      if (onProgress && e.lengthComputable) onProgress(e.loaded, e.total);
    });
    xhr.addEventListener("load", () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      reject(new Error((data && data.detail) || `업로드 실패 (HTTP ${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("네트워크 오류로 업로드에 실패했습니다.")));
    xhr.addEventListener("abort", () => reject(new Error("업로드가 취소되었습니다.")));
    xhr.send(fd);
  });
}

/** 보고서를 내려받는다. 저장 위치는 브라우저가 묻거나 다운로드 폴더로 간다.
 *  응답 헤더의 파일명을 그대로 쓰고, 실패하면 서버 메시지를 그대로 보여준다. */
async function downloadFile(path, fallbackName) {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    let msg = `보고서 생성 실패 (HTTP ${res.status})`;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch (_) {}
    throw new Error(msg);
  }

  // Content-Disposition에서 파일명 추출 (한글은 filename*=utf-8'' 형식으로 온다)
  const cd = res.headers.get("content-disposition") || "";
  let name = fallbackName;
  const star = cd.match(/filename\*=utf-8''([^;]+)/i);
  const plain = cd.match(/filename="?([^";]+)"?/i);
  if (star) name = decodeURIComponent(star[1]);
  else if (plain) name = plain[1];

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return name;
}

export const api = {
  getQueue: () => req("GET", "/api/queue"),
  addVideos: (paths) => req("POST", "/api/queue/add", { paths }),
  uploadVideos,
  clearQueue: () => req("POST", "/api/queue/clear"),
  selectVideo: (name) => req("POST", "/api/queue/select", { name }),

  runAnalysis: () => req("POST", "/api/analysis/run"),

  getResults: () => req("GET", "/api/results"),
  setSiteName: (value) => req("POST", "/api/results/site_name", { value }),
  setPipeCondition: (value) => req("POST", "/api/results/pipe_condition", { value }),
  editRow: (video, time_s, field, value) =>
    req("PATCH", "/api/results/row", { video, time_s, field, value }),
  addManualRow: (video, time_s) =>
    req("POST", "/api/results/manual_row", { video, time_s }),
  deleteRow: (video, time_s) => req("DELETE", "/api/results/row", { video, time_s }),
  deleteGroup: (video, dist) => req("DELETE", "/api/results/group", { video, dist }),

  exportExcel: (path) => req("POST", "/api/export/excel", { path }),
  downloadExcel: () => downloadFile("/api/export/excel/download", "CCTV조사표.xlsx"),
  downloadPipeassetPdf: () => downloadFile("/api/export/pdf/download", "CCTV야장.pdf"),

  getReportMeta: (video) =>
    req("GET", "/api/config/report_meta" + (video ? `?video=${encodeURIComponent(video)}` : "")),
  setReportMeta: (values, video) => req("POST", "/api/config/report_meta", { values, video }),

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
