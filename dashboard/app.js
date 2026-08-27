const canvas = document.getElementById("grid");
const ctx = canvas.getContext("2d");

function color(ref) {
  if (ref <= 0) return "#1b2430";
  if (ref === 1) return "#3d8bfd";
  if (ref === 2) return "#f0b429";
  return "#e85d4c";
}

function draw(blocks) {
  const n = blocks.length || 1;
  const cols = Math.ceil(Math.sqrt(n * 1.8));
  const rows = Math.ceil(n / cols);
  const pad = 2;
  const cw = canvas.width;
  const ch = canvas.height;
  const tw = (cw - pad) / cols;
  const th = (ch - pad) / rows;
  ctx.clearRect(0, 0, cw, ch);
  blocks.forEach((b, i) => {
    const x = (i % cols) * tw + pad;
    const y = Math.floor(i / cols) * th + pad;
    ctx.fillStyle = color(b.ref);
    ctx.fillRect(x, y, Math.max(1, tw - pad), Math.max(1, th - pad));
  });
}

function set(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/metrics`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    const blocks = data.blocks || [];
    draw(blocks);
    const m = data.metrics || {};
    set("kv", m.kv_utilization != null ? `${(m.kv_utilization * 100).toFixed(1)}%` : "—");
    set("hit", m.hit_rate != null ? `${(m.hit_rate * 100).toFixed(1)}%` : "—");
    set("run", m.running != null ? String(m.running) : "—");
    set("wait", m.waiting != null ? String(m.waiting) : "—");
    set("nblocks", String(blocks.length));
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

connect();
