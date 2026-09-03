import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

// Numbers below are verbatim from reports/ (frozen results).

// ---- interval chart: between vs within pad, two exposures ----
function drawContrast() {
  const svg = document.getElementById("contrast");
  const W = 820, H = 200, m = { l: 300, r: 40 };
  const rows = [
    { label: "offset withdrawal · between pads", v: 1.9, lo: null, hi: null, c: "#898781", y: 40 },
    { label: "offset withdrawal · WITHIN pads", v: -5.3, lo: -11.4, hi: -0.5, c: "#d7301f", y: 75 },
    { label: "offset count · between pads", v: 0.9, lo: null, hi: null, c: "#898781", y: 125 },
    { label: "offset count · WITHIN pads", v: -2.8, lo: -9.6, hi: 1.8, c: "#ef6548", y: 160 },
  ];
  const lo = -13, hi = 4;
  const x = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
  let s = `<line x1="${x(0)}" x2="${x(0)}" y1="20" y2="${H - 20}"
    stroke="#c3c2b7" stroke-dasharray="3 3"/>
    <text x="${x(0)}" y="14" text-anchor="middle" font-size="10"
    fill="#898781">0 (no effect)</text>`;
  for (const r of rows) {
    s += `<text x="${m.l - 10}" y="${r.y + 4}" text-anchor="end"
      font-size="12" fill="#52514e">${r.label}</text>`;
    if (r.lo !== null)
      s += `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${r.y}" y2="${r.y}"
        stroke="${r.c}" stroke-width="2.5" opacity="0.55"/>`;
    s += `<circle cx="${x(r.v)}" cy="${r.y}" r="6" fill="${r.c}"/>
      <text x="${x(r.v)}" y="${r.y - 11}" text-anchor="middle"
      font-size="12" font-weight="700" fill="${r.c}">${
        r.v > 0 ? "+" + r.v : r.v}</text>`;
  }
  svg.innerHTML = s;
}

// ---- depletion penalty by crowding ----
function drawPenalty() {
  const svg = document.getElementById("penalty");
  const W = 700, H = 220, m = { t: 26, b: 40, l: 10, r: 10 };
  const bins = [
    { label: "1–3 offsets", v: -23.2, n: 458 },
    { label: "4–8", v: -49.4, n: 936 },
    { label: "9–12", v: -66.1, n: 828 },
    { label: ">12", v: -93.5, n: 8115 },
  ];
  const bw = (W - m.l - m.r) / bins.length;
  const y = (v) => m.t + (v / -100) * (H - m.t - m.b);
  let s = "";
  bins.forEach((b, i) => {
    const xx = m.l + i * bw;
    s += `<rect x="${xx + 20}" y="${y(0)}" width="${bw - 40}"
      height="${y(b.v) - y(0)}" rx="4" fill="#d7301f" opacity="0.85">
      <title>${b.label}: median ${b.v} pts (n=${b.n.toLocaleString()})</title></rect>
      <text x="${xx + bw / 2}" y="${y(b.v) + 16}" text-anchor="middle"
      font-size="12" font-weight="700" fill="#0b0b0b">${b.v}</text>
      <text x="${xx + bw / 2}" y="${H - 22}" text-anchor="middle"
      font-size="11" fill="#52514e">${b.label}</text>
      <text x="${xx + bw / 2}" y="${H - 9}" text-anchor="middle"
      font-size="9" fill="#898781">n=${b.n.toLocaleString()}</text>`;
  });
  s += `<text x="${m.l + 4}" y="${m.t - 10}" font-size="10"
    fill="#898781">median penalty, pts of peer-P50 pace (offsets' withdrawal erased)</text>`;
  svg.innerHTML = s;
}

// ---- the laterals map ----
const map = new maplibregl.Map({
  container: "map",
  style: "https://tiles.openfreemap.org/styles/positron",
  bounds: [
    [-120.5, 49.0],
    [-110.0, 58.5],
  ],
  fitBoundsOptions: { padding: 14 },
  attributionControl: { compact: true },
  cooperativeGestures: true,
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
map.on("error", (e) => console.error("[map]", e.error ?? e));
window._map = map;

map.on("load", async () => {
  const legs = await fetch("/data/legs.json").then((r) => r.json());
  map.addSource("legs", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: legs.map(([hlon, hlat, tlon, tlat]) => ({
        type: "Feature",
        geometry: { type: "LineString",
          coordinates: [[hlon, hlat], [tlon, tlat]] },
        properties: {},
      })),
    },
  });
  map.addLayer({
    id: "legs",
    type: "line",
    source: "legs",
    paint: {
      "line-color": "#256abf",
      "line-width": ["interpolate", ["linear"], ["zoom"],
        4, 0.4, 8, 0.9, 12, 2.2],
      "line-opacity": ["interpolate", ["linear"], ["zoom"],
        4, 0.35, 8, 0.6, 12, 0.9],
    },
  });
});

// ---- real fishbone in 3-D (licence 0318429, from ST37 surveys) ----
async function initFishbone() {
  const box = document.getElementById("fish-box");
  const canvas = document.getElementById("fishbone");
  if (!box || !canvas) return;
  const { legs } = await fetch("/data/fishbone.json").then((r) => r.json());

  const VE = 2; // vertical exaggeration (stated in the caption)
  // world: x east, y north (metres, local to surface hole), z up
  const pts3 = legs.map((l) => l.pts.map(([x, y, d]) => [x, y, -d * VE]));
  const all = pts3.flat();
  const min = [0, 1, 2].map((i) => Math.min(...all.map((p) => p[i])));
  const max = [0, 1, 2].map((i) => Math.max(...all.map((p) => p[i])));
  const c = [0, 1, 2].map((i) => (min[i] + max[i]) / 2);
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  // landing plane sits 25 m (true) below the deepest station so the drop
  // ticks trace each leg's undulation instead of vanishing into the plane
  const zPlane = min[2] - 25 * VE;

  let yaw = 0.9, pitch = 1.1, dragging = false, auto = true, px = 0, py = 0;

  function project(p, W, H, s) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const ct = Math.cos(pitch), st = Math.sin(pitch);
    const x1 = (p[0] - c[0]) * cy - (p[1] - c[1]) * sy;
    const y1 = (p[0] - c[0]) * sy + (p[1] - c[1]) * cy;
    const z1 = p[2] - c[2];
    // tilt about the screen-x axis; z2 is "up on screen"
    const z2 = y1 * st + z1 * ct;
    return [W / 2 + x1 * s, H / 2 - z2 * s];
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const W = box.clientWidth, H = box.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const s = (Math.min(W, H) * 0.82) / span;
    const P = (p) => project(p, W, H, s);

    // landing plane: grid every 250 m over the xy bounds
    const pad = 120;
    const [gx0, gy0] = [min[0] - pad, min[1] - pad];
    const [gx1, gy1] = [max[0] + pad, max[1] + pad];
    const corners = [
      [gx0, gy0, zPlane], [gx1, gy0, zPlane],
      [gx1, gy1, zPlane], [gx0, gy1, zPlane],
    ].map(P);
    g.beginPath();
    corners.forEach((q, i) => (i ? g.lineTo(...q) : g.moveTo(...q)));
    g.closePath();
    g.fillStyle = "rgba(143, 160, 138, 0.16)";
    g.fill();
    g.strokeStyle = "rgba(11,11,11,0.10)";
    g.lineWidth = 1;
    for (let x = Math.ceil(gx0 / 250) * 250; x <= gx1; x += 250) {
      g.beginPath(); g.moveTo(...P([x, gy0, zPlane])); g.lineTo(...P([x, gy1, zPlane])); g.stroke();
    }
    for (let y = Math.ceil(gy0 / 250) * 250; y <= gy1; y += 250) {
      g.beginPath(); g.moveTo(...P([gx0, y, zPlane])); g.lineTo(...P([gx1, y, zPlane])); g.stroke();
    }

    // drop ticks from the laterals to the landing plane (every 6th station,
    // laterals only — deeper than 80% of max depth)
    g.strokeStyle = "rgba(90, 130, 82, 0.45)";
    for (const leg of pts3)
      for (let i = 0; i < leg.length; i += 6) {
        const p = leg[i];
        if (p[2] > zPlane * 0.8) continue;
        g.beginPath(); g.moveTo(...P(p)); g.lineTo(...P([p[0], p[1], zPlane])); g.stroke();
      }

    // the legs
    g.lineJoin = "round"; g.lineCap = "round";
    g.strokeStyle = "rgba(37, 106, 191, 0.85)";
    g.lineWidth = 2;
    for (const leg of pts3) {
      g.beginPath();
      leg.forEach((p, i) => (i ? g.lineTo(...P(p)) : g.moveTo(...P(p))));
      g.stroke();
    }
    // surface hole marker
    const sh = P(pts3[0][0]);
    g.fillStyle = "#d95f00";
    g.beginPath(); g.arc(sh[0], sh[1], 4, 0, 7); g.fill();
    g.fillStyle = "#52514e"; g.font = "11px system-ui, sans-serif";
    g.fillText("surface hole", sh[0] + 8, sh[1] + 4);

    // screen-space scale bar (orthographic: 500 m east-west = 500*s px)
    g.strokeStyle = "#52514e"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(12, H - 30); g.lineTo(12 + 500 * s, H - 30); g.stroke();
    g.fillStyle = "#52514e";
    g.fillText("500 m", 12 + 250 * s - 16, H - 36);
    g.fillStyle = "#898781";
    g.fillText(`laterals land at TVD ≈ ${Math.round(-min[2] / VE)} m`, 12, H - 12);
  }

  function frame() {
    if (auto) { yaw += 0.0025; draw(); }
    requestAnimationFrame(frame);
  }
  box.addEventListener("pointerdown", (e) => {
    dragging = true; auto = false; px = e.clientX; py = e.clientY;
    box.setPointerCapture(e.pointerId); box.style.cursor = "grabbing";
  });
  box.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    yaw += (e.clientX - px) * 0.008;
    pitch = Math.min(1.45, Math.max(0.25, pitch + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY; draw();
  });
  box.addEventListener("pointerup", () => { dragging = false; box.style.cursor = "grab"; });
  new ResizeObserver(draw).observe(box);
  draw();
  frame();
}

drawContrast();
drawPenalty();
initFishbone();
