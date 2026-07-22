const handlers = {
  log: [],
  progress: [],
  result_update: [],
  batch_done: [],
};

export function on(type, fn) {
  handlers[type].push(fn);
}

function dispatch(msg) {
  const list = handlers[msg.type];
  if (list) list.forEach((fn) => fn(msg));
}

export function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/progress`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      dispatch(JSON.parse(ev.data));
    } catch (_) {}
  };
  ws.onclose = () => {
    setTimeout(connectWs, 1500);
  };
  return ws;
}
