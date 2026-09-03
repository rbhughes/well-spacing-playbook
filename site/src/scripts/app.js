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

drawContrast();
drawPenalty();
