import {
  FaceLandmarker,
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";

const CANVAS = 512;
const ED = 96;
const EYE_Y = 230;
const EX = 256;
const BG = [28, 30, 32];
const L_IRIS = [EX - ED / 2, EYE_Y];
const R_IRIS = [EX + ED / 2, EYE_Y];
const RAD = Math.PI / 180;
const MAX_YAW = 0.5;   // strongly turned heads are rejected outright
const HEAD_YAW = 0.22; // the base head must be close to frontal

// Paste order: back to front.
const SLOTS = ["neck", "hair", "ears", "eyes", "brows", "nose", "chin", "mouth"];

// Slots whose base-head counterpart is erased (smooth skin fill) before the donor is pasted.
const FACE_ERASE = new Set(["eyes", "brows", "nose", "mouth", "chin"]);

// MediaPipe face-oval loop: the "head" layer is only this oval (skin, cheeks, jaw line);
// everything outside it - hair, ears, neck and shoulders - comes from the other donors.
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const CLIP_GROW = { ears: 1.4, eyes: 1.35, brows: 1.45, nose: 1.6, chin: 1.5, mouth: 1.5 };
const MASK_GROW = { ears: 1.15, eyes: 1.2, brows: 1.2, nose: 1.15, chin: 1.1, mouth: 1.08 };
// Parts of the donor that must not travel along with the cut-out.
const SUBTRACT = {
  eyes: { layer: "brows", grow: 1.15 },
  chin: { layer: "mouth", grow: 1.1 },
  mouth: { layer: "nose", grow: 1.0 },
  hair: { layer: "brows", grow: 1.3 },
};

// Placement = similarity transform only (uniform scale + small rotation + translation).
// pivot: landmark aligned exactly; axis: landmark pair giving scale and rotation;
// chk: landmarks whose residual measures how well the proportions agree;
// s / r: allowed scale range and max rotation (degrees); wy: weight of the head-direction mismatch.
const FIT = {
  eyes:  { pivot: "eye_mid",   axis: ["il", "ir"],           chk: [],                   s: [0.92, 1.1],  r: 5, wy: 2.0 },
  brows: { pivot: "brow_mid",  axis: ["brow_l", "brow_r"],   chk: ["eye_mid"],          s: [0.85, 1.2],  r: 8, wy: 1.0 },
  nose:  { pivot: "nose",      axis: ["eye_mid", "nose"],    chk: ["mouth_mid"],        s: [0.85, 1.2],  r: 8, wy: 3.0 },
  chin:  { pivot: "chin",      axis: ["mouth_mid", "chin"],  chk: ["jaw_l", "jaw_r"],   s: [0.85, 1.2],  r: 8, wy: 1.5 },
  mouth: { pivot: "mouth_mid", axis: ["mouth_l", "mouth_r"], chk: ["nose", "chin"],     s: [0.85, 1.25], r: 8, wy: 2.5 },
  hair:  { pivot: "top",       axis: ["face_l", "face_r"],   chk: ["eye_mid"],          s: [0.85, 1.25], r: 6, wy: 0.8 },
  neck:  { pivot: "chin",      axis: ["face_l", "face_r"],   chk: ["jaw_l", "jaw_r"],   s: [0.8, 1.3],   r: 6, wy: 1.0 },
  ears:  { pivot: "face_mid",  axis: ["face_l", "face_r"],   chk: ["eye_mid", "chin"],  s: [0.85, 1.2],  r: 6, wy: 1.0 },
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function newCanvas(w = CANVAS, h = CANVAS) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function cloneCanvas(c) {
  const t = newCanvas(c.width, c.height);
  t.getContext("2d").drawImage(c, 0, 0);
  return t;
}

function applyM(pts, m) {
  return pts.map(([x, y]) => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]]);
}

function setTransform(x, m) {
  x.setTransform(m[0], m[3], m[1], m[4], m[2], m[5]);
}

function threshold(c) {
  const x = c.getContext("2d", { willReadFrequently: true });
  const im = x.getImageData(0, 0, c.width, c.height);
  const d = im.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i + 3] = d[i + 3] > 127 ? 255 : 0;
  }
  x.putImageData(im, 0, 0);
  return c;
}

// ---------------------------------------------------------------- geometry

const avg = (...ps) => [
  ps.reduce((s, p) => s + p[0], 0) / ps.length,
  ps.reduce((s, p) => s + p[1], 0) / ps.length,
];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

function geom(tlm) {
  const g = {
    il: tlm[468], ir: tlm[473], eye_mid: mid(tlm[468], tlm[473]),
    ed: dist(tlm[468], tlm[473]),
    nose: tlm[1], mouth_l: tlm[61], mouth_r: tlm[291],
    mouth_mid: mid(tlm[61], tlm[291]), chin: tlm[152],
    // centres of the two eyebrows (average of the upper and lower mesh rows)
    brow_l: avg(tlm[105], tlm[66], tlm[52], tlm[65]),
    brow_r: avg(tlm[334], tlm[296], tlm[282], tlm[295]),
    face_l: tlm[234], face_r: tlm[454], face_mid: mid(tlm[234], tlm[454]),
    top: tlm[10], jaw_l: tlm[172], jaw_r: tlm[397],
  };
  g.brow_mid = mid(g.brow_l, g.brow_r);
  g.fw = dist(g.face_r, g.face_l);
  // signed head direction: >0 nose points towards the viewer's right
  g.yaw = (2 * g.nose[0] - g.face_l[0] - g.face_r[0]) / g.fw;
  return g;
}

// Geometry of the horizontally mirrored face. Left/right labels are swapped so that
// "mouth_l" is again the corner on the viewer's left - this is what makes a mirrored
// donor genuinely different from the unmirrored one.
function mirrorGeom(g) {
  const M = (p) => [CANVAS - p[0], p[1]];
  const m = {
    il: M(g.ir), ir: M(g.il), nose: M(g.nose),
    mouth_l: M(g.mouth_r), mouth_r: M(g.mouth_l), chin: M(g.chin),
    brow_l: M(g.brow_r), brow_r: M(g.brow_l),
    face_l: M(g.face_r), face_r: M(g.face_l), top: M(g.top),
    jaw_l: M(g.jaw_r), jaw_r: M(g.jaw_l), ed: g.ed, fw: g.fw,
  };
  m.eye_mid = mid(m.il, m.ir);
  m.mouth_mid = mid(m.mouth_l, m.mouth_r);
  m.brow_mid = mid(m.brow_l, m.brow_r);
  m.face_mid = mid(m.face_l, m.face_r);
  m.yaw = -g.yaw;
  return m;
}

function specs(layer, g, grow = 1) {
  const ed = g.ed;
  const fw = dist(g.face_r, g.face_l);
  const P = (x, y, rx, ry) => ({ x, y, rx: rx * grow, ry: ry * grow });
  switch (layer) {
    case "neck":
      return [P(g.face_mid[0], g.chin[1] + 1.4 * ed, 0.95 * fw, 1.45 * ed)];
    case "ears": {
      // ear-shaped (tall) ellipses centred just outside the face edge
      const ey = g.eye_mid[1] + 0.25 * ed;
      const side = 0.5 * fw + 0.12 * ed;
      return [P(g.face_mid[0] - side, ey, 0.2 * ed, 0.45 * ed),
              P(g.face_mid[0] + side, ey, 0.2 * ed, 0.45 * ed)];
    }
    case "eyes":
      return [P(g.il[0], g.il[1] + 0.02 * ed, 0.3 * ed, 0.2 * ed),
              P(g.ir[0], g.ir[1] + 0.02 * ed, 0.3 * ed, 0.2 * ed)];
    case "brows": {
      const sp = dist(g.brow_r, g.brow_l);
      return [P(g.brow_l[0], g.brow_l[1], 0.4 * sp, 0.13 * ed),
              P(g.brow_r[0], g.brow_r[1], 0.4 * sp, 0.13 * ed)];
    }
    case "nose": {
      const nl = dist(g.nose, g.eye_mid);
      const c = [g.eye_mid[0] + 0.62 * (g.nose[0] - g.eye_mid[0]),
                 g.eye_mid[1] + 0.62 * (g.nose[1] - g.eye_mid[1])];
      return [P(c[0], c[1], 0.44 * nl, 0.72 * nl)];
    }
    case "chin": {
      const mw = dist(g.mouth_r, g.mouth_l);
      const depth = Math.max(g.chin[1] - g.mouth_mid[1], 0.3 * ed);
      return [P(g.face_mid[0], g.mouth_mid[1] + 0.8 * depth, 0.6 * mw, 0.55 * depth)];
    }
    case "mouth": {
      const mw = dist(g.mouth_r, g.mouth_l);
      return [P(g.mouth_mid[0], g.mouth_mid[1], 0.72 * mw, 0.55 * mw)];
    }
    default:
      return [];
  }
}

// ---------------------------------------------------------------- masks

function blobPath(x, cx, cy, rx, ry, n = 18) {
  x.beginPath();
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    const j = 1 + (Math.random() * 0.21 - 0.09);
    const px = cx + rx * j * Math.cos(th);
    const py = cy + ry * j * Math.sin(th);
    if (i === 0) x.moveTo(px, py); else x.lineTo(px, py);
  }
  x.closePath();
  x.fill();
}

function ellipsePath(x, cx, cy, rx, ry, A = null, n = 40) {
  x.beginPath();
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    let px = cx + rx * Math.cos(th);
    let py = cy + ry * Math.sin(th);
    if (A) {
      const qx = A[0] * px + A[1] * py + A[2];
      py = A[3] * px + A[4] * py + A[5];
      px = qx;
    }
    if (i === 0) x.moveTo(px, py); else x.lineTo(px, py);
  }
  x.closePath();
  x.fill();
}

function smoothMask(c) {
  const t = newCanvas(c.width, c.height);
  const x = t.getContext("2d", { willReadFrequently: true });
  x.filter = "blur(1.4px)";
  x.drawImage(c, 0, 0);
  x.filter = "none";
  return threshold(t);
}

function compIn(c, other) {
  const x = c.getContext("2d", { willReadFrequently: true });
  x.globalCompositeOperation = "destination-in";
  x.drawImage(other, 0, 0);
  x.globalCompositeOperation = "source-over";
  return c;
}

function destIn(c, other) {
  compIn(c, other);
  return threshold(c);
}

function destOut(c, other) {
  const x = c.getContext("2d", { willReadFrequently: true });
  x.globalCompositeOperation = "destination-out";
  x.drawImage(other, 0, 0);
  x.globalCompositeOperation = "source-over";
  return threshold(c);
}

function aa(c) {
  const t = newCanvas(c.width, c.height);
  const x = t.getContext("2d", { willReadFrequently: true });
  x.filter = "blur(0.9px)";
  x.drawImage(c, 0, 0);
  x.filter = "none";
  return t;
}

function featureMask(layer, g, sil, grow = 1) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  for (const s of specs(layer, g, grow)) blobPath(x, s.x, s.y, s.rx, s.ry);
  threshold(c);
  if (sil) destIn(c, sil);
  return threshold(c);
}

// Face oval as a point list (optionally scaled about its centre).
function ovalPoints(tlm, scale = 1) {
  const pts = OVAL.map((i) => tlm[i]);
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return pts.map(([x, y]) => [cx + scale * (x - cx), cy + scale * (y - cy)]);
}

function pointInPoly(pts, x, y) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i];
    const [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// Smooth closed curve through the polygon (midpoint quadratic splines).
function polyPath(x, pts) {
  const n = pts.length;
  const m = (i) => [(pts[i][0] + pts[(i + 1) % n][0]) / 2, (pts[i][1] + pts[(i + 1) % n][1]) / 2];
  const s = m(n - 1);
  x.beginPath();
  x.moveTo(s[0], s[1]);
  for (let i = 0; i < n; i++) {
    const e = m(i);
    x.quadraticCurveTo(pts[i][0], pts[i][1], e[0], e[1]);
  }
  x.closePath();
}

function polyCanvas(pts) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  polyPath(x, pts);
  x.fill();
  return threshold(c);
}

// Hair = everything above the brow line (forehead + hair volume) plus, beside the face,
// the band outside the face oval down to the jaw corners (the neck region takes over there).
function hairRegion(g, ovalC) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  const cut = Math.max(0, g.eye_mid[1] - 0.32 * g.ed);
  const low = Math.min(g.jaw_l[1], g.jaw_r[1]) + 0.05 * g.ed;
  x.fillRect(0, 0, CANVAS, cut);
  const band = newCanvas();
  const bx = band.getContext("2d", { willReadFrequently: true });
  bx.fillStyle = "#fff";
  bx.fillRect(0, cut, CANVAS, Math.max(0, low - cut));
  destOut(band, ovalC);
  x.drawImage(band, 0, 0);
  return threshold(c);
}

// Neck and shoulders = everything below the jaw corners that is outside the face oval.
function neckRegion(g, ovalC) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  const y0 = Math.min(g.jaw_l[1], g.jaw_r[1]);
  x.fillRect(0, y0, CANVAS, CANVAS - y0);
  return destOut(c, ovalC);
}

// Donor-side ellipses, transformed into head coordinates by A.
function ellipseMask(g, A, layer, grow) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  for (const s of specs(layer, g, grow)) ellipsePath(x, s.x, s.y, s.rx, s.ry, A);
  return threshold(c);
}

function warpAlpha(alphaCanvas, A, fx) {
  return threshold(drawWarped(alphaCanvas, A, fx, false));
}

function drawWarped(imgCanvas, A, fx, fill = true) {
  const c = newCanvas();
  const x = c.getContext("2d");
  if (fill) {
    x.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
    x.fillRect(0, 0, CANVAS, CANVAS);
  }
  x.imageSmoothingEnabled = true;
  x.imageSmoothingQuality = "high";
  x.setTransform(A[0], A[3], A[1], A[4], A[2], A[5]);
  if (fx) {
    x.translate(CANVAS, 0);
    x.scale(-1, 1);
  }
  x.drawImage(imgCanvas, 0, 0);
  x.setTransform(1, 0, 0, 1, 0, 0);
  return c;
}

// Replace the head's own features by a smooth fill sampled from the surrounding skin,
// so that nothing of the original eyes/nose/mouth can show through a donor patch.
function eraseFill(hp, er, hg) {
  const known = newCanvas();
  const kx = known.getContext("2d", { willReadFrequently: true });
  kx.drawImage(hp, 0, 0);
  kx.globalCompositeOperation = "destination-out";
  kx.drawImage(er, 0, 0);
  kx.globalCompositeOperation = "source-over";

  const x0 = Math.max(0, Math.round(hg.face_mid[0] - 0.4 * hg.fw));
  const x1 = Math.min(CANVAS, Math.round(hg.face_mid[0] + 0.4 * hg.fw));
  const y0 = Math.max(0, Math.round(hg.eye_mid[1] - 0.1 * hg.ed));
  const y1 = Math.min(CANVAS, Math.round(hg.chin[1] - 0.2 * hg.ed));
  let flat = "rgb(205,165,145)";
  if (x1 > x0 && y1 > y0) {
    const d = kx.getImageData(x0, y0, x1 - x0, y1 - y0).data;
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] > 200) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
    }
    if (n) flat = `rgb(${(r / n) | 0},${(g / n) | 0},${(b / n) | 0})`;
  }

  const fill = newCanvas();
  const fx = fill.getContext("2d");
  fx.fillStyle = flat;
  fx.fillRect(0, 0, CANVAS, CANVAS);
  for (const rad of [64, 32, 16, 8, 4]) {
    fx.filter = `blur(${rad}px)`;
    fx.drawImage(known, 0, 0);
  }
  fx.filter = "none";
  fx.drawImage(known, 0, 0);
  compIn(fill, er);
  hp.getContext("2d").drawImage(fill, 0, 0);
}

// ---------------------------------------------------------------- fitting

const RING = (() => {
  const p = [[0, 0]];
  for (let i = 0; i < 8; i++) {
    const t = (i * Math.PI) / 4;
    p.push([0.5 * Math.cos(t), 0.5 * Math.sin(t)]);
    p.push([0.9 * Math.cos(t), 0.9 * Math.sin(t)]);
  }
  return p;
})();

function inAlpha(face, x, y) {
  const ix = Math.floor(x / 8);
  const iy = Math.floor(y / 8);
  return ix >= 0 && ix < 64 && iy >= 0 && iy < 64 && face.a64[iy * 64 + ix] === 1;
}

// Sample points of the head-side region that the donor has to cover.
function slotPoints(layer, hg, head) {
  const oval = head.oval || (head.oval = ovalPoints(head.tlm, 1));
  const pts = [];
  if (layer === "hair") {
    // the hair volume right above the forehead
    const cut = hg.eye_mid[1] - 0.32 * hg.ed;
    const y0 = hg.top[1] - 0.3 * hg.ed;
    for (let j = 0; j < 4; j++) {
      for (let i = 0; i < 7; i++) {
        pts.push([hg.face_mid[0] + (i / 6 - 0.5) * 0.8 * hg.fw, y0 + (j / 3) * (cut - y0)]);
      }
    }
  } else if (layer === "neck") {
    // the neck / shoulder column just below the chin
    const y0 = hg.chin[1] + 0.1 * hg.ed;
    for (let j = 0; j < 4; j++) {
      for (let i = 0; i < 7; i++) {
        pts.push([hg.face_mid[0] + (i / 6 - 0.5) * 0.7 * hg.fw, y0 + (j / 3) * 0.5 * hg.ed]);
      }
    }
  } else {
    for (const s of specs(layer, hg)) {
      for (const [u, v] of RING) {
        const x = s.x + u * s.rx;
        const y = s.y + v * s.ry;
        // ears are only pasted outside the face oval
        if (layer === "ears" && pointInPoly(oval, x, y)) continue;
        pts.push([x, y]);
      }
    }
  }
  return pts;
}

// Fraction of the head-side sample points that land on the donor's silhouette.
function coverage(face, A, flip, pts) {
  if (!pts.length) return 1;
  const det = A[0] * A[4] - A[1] * A[3];
  if (Math.abs(det) < 1e-9) return 0;
  const ia = A[4] / det, ib = -A[1] / det, id = -A[3] / det, ie = A[0] / det;
  let n = 0;
  for (const [X, Y] of pts) {
    const dx = X - A[2], dy = Y - A[5];
    let x = ia * dx + ib * dy;
    const y = id * dx + ie * dy;
    if (flip) x = CANVAS - x;
    if (inAlpha(face, x, y)) n++;
  }
  return n / pts.length;
}

function fitSlot(layer, hg, face, flip, pts) {
  const F = FIT[layer];
  const gd = flip ? face.gm : face.g;
  const vb = [hg[F.axis[1]][0] - hg[F.axis[0]][0], hg[F.axis[1]][1] - hg[F.axis[0]][1]];
  const vd = [gd[F.axis[1]][0] - gd[F.axis[0]][0], gd[F.axis[1]][1] - gd[F.axis[0]][1]];
  const nb = Math.hypot(vb[0], vb[1]);
  const nd = Math.hypot(vd[0], vd[1]);
  if (nb < 1e-6 || nd < 1e-6) return null;
  const sRaw = nb / nd;
  let thRaw = Math.atan2(vb[1], vb[0]) - Math.atan2(vd[1], vd[0]);
  thRaw = Math.atan2(Math.sin(thRaw), Math.cos(thRaw));
  const s = clamp(sRaw, F.s[0], F.s[1]);
  const th = clamp(thRaw, -F.r * RAD, F.r * RAD);
  const c = Math.cos(th) * s;
  const sn = Math.sin(th) * s;
  const pb = hg[F.pivot];
  const qd = gd[F.pivot];
  const A = [c, -sn, pb[0] - (c * qd[0] - sn * qd[1]),
             sn, c, pb[1] - (sn * qd[0] + c * qd[1])];
  let chk = 0;
  for (const k of F.chk) {
    const q = gd[k];
    const x = A[0] * q[0] + A[1] * q[1] + A[2];
    const y = A[3] * q[0] + A[4] * q[1] + A[5];
    chk += Math.hypot(x - hg[k][0], y - hg[k][1]);
  }
  chk = F.chk.length ? chk / (F.chk.length * hg.ed) : 0;
  const cov = coverage(face, A, flip, pts);
  const yawd = Math.abs(gd.yaw - hg.yaw);
  let cost = 2.5 * (1 - cov) + 3.0 * chk + 1.8 * Math.abs(Math.log(sRaw))
    + 0.8 * (Math.abs(thRaw) / (10 * RAD)) + F.wy * yawd + (flip ? 0.03 : 0);
  // ears: prefer donors whose ears are actually visible skin rather than hidden under hair
  if (layer === "ears") cost += 4.0 * face.earDist;
  return { A, flip, cost, cov, chk, sRaw, thRaw, s, th, yawd };
}

function bestFit(layer, hg, face, pts) {
  let best = null;
  for (const flip of [false, true]) {
    const f = fitSlot(layer, hg, face, flip, pts);
    if (f && (!best || f.cost < best.cost)) best = f;
  }
  return best || { A: [1, 0, 0, 0, 1, 0], flip: false, cost: 9, cov: 0, chk: 0,
                   sRaw: 1, thRaw: 0, s: 1, th: 0, yawd: 0 };
}

// Optimal one-to-one assignment (Kuhn-Munkres), rows <= columns.
function hungarian(a) {
  const n = a.length, m = a[0].length;
  const INF = 1e9;
  const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0);
  const p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(m + 1).fill(INF);
    const used = new Array(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const ans = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) ans[p[j] - 1] = j - 1;
  return ans;
}

function assign(head, others) {
  const hg = head.g;
  const pts = {};
  for (const s of SLOTS) pts[s] = slotPoints(s, hg, head);
  const fits = SLOTS.map((s) => others.map((f) => bestFit(s, hg, f, pts[s])));
  const cost = fits.map((row) => row.map((x) => x.cost + Math.random() * 0.04));
  const cols = hungarian(cost);
  const picks = cols.map((j, i) => ({ layer: SLOTS[i], face: others[j], fit: fits[i][j] }));
  const total = picks.reduce((t, p) => t + p.fit.cost, 0);
  return { head, picks, total };
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const k = (Math.random() * (i + 1)) | 0;
    [a[i], a[k]] = [a[k], a[i]];
  }
  return a;
}

// Choose the head and give every slot a different donor, minimising the total misfit.
function planCollage(pool) {
  let heads = pool.filter((f) => Math.abs(f.g.yaw) <= HEAD_YAW);
  if (heads.length < 2) {
    heads = pool.slice().sort((a, b) => Math.abs(a.g.yaw) - Math.abs(b.g.yaw)).slice(0, 2);
  }
  const cands = shuffle(heads.slice()).slice(0, 4);
  let best = null;
  for (const head of cands) {
    const plan = assign(head, pool.filter((f) => f !== head));
    if (!best || plan.total < best.total) best = plan;
  }
  return best;
}

// ---------------------------------------------------------------- composite

function composite(plan) {
  const out = newCanvas();
  const ox = out.getContext("2d");
  ox.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
  ox.fillRect(0, 0, CANVAS, CANVAS);

  const head = plan.head;
  const hg = head.g;
  const sil = head.alpha;
  const pick = {};
  for (const p of plan.picks) pick[p.layer] = p;

  // The head is only its face oval; hair, ears, neck and shoulders belong to other donors.
  const ovalC = polyCanvas(ovalPoints(head.tlm, 1));
  const headClip = polyCanvas(ovalPoints(head.tlm, 1.04));
  const hp = newCanvas();
  hp.getContext("2d").drawImage(head.warped, 0, 0);
  compIn(hp, aa(destIn(cloneCanvas(sil), headClip)));

  // Hard-edged cut regions on the head; the very same shapes are erased and re-filled.
  const masks = {};
  for (const layer of FACE_ERASE) masks[layer] = featureMask(layer, hg, sil, MASK_GROW[layer]);
  const er = newCanvas();
  const ex = er.getContext("2d");
  for (const layer of FACE_ERASE) ex.drawImage(masks[layer], 0, 0);
  eraseFill(hp, er, hg);

  // Cut a donor's pixels through mask m and paste them with a clean vector-like edge.
  const paste = (layer, m) => {
    const { face: donor, fit } = pick[layer];
    const gd = fit.flip ? donor.gm : donor.g;
    if (CLIP_GROW[layer]) m = destIn(m, ellipseMask(gd, fit.A, layer, CLIP_GROW[layer]));
    const sub = SUBTRACT[layer];
    if (sub) m = destOut(m, ellipseMask(gd, fit.A, sub.layer, sub.grow));
    m = destIn(m, warpAlpha(donor.alpha, fit.A, fit.flip));
    m = smoothMask(m);
    const patch = drawWarped(donor.warped, fit.A, fit.flip);
    compIn(patch, aa(m));
    ox.drawImage(patch, 0, 0);
  };

  paste("neck", neckRegion(hg, ovalC));        // behind the chin
  ox.drawImage(hp, 0, 0);                      // the face oval
  paste("hair", hairRegion(hg, ovalC));
  paste("ears", destOut(featureMask("ears", hg, null, MASK_GROW.ears), ovalC));
  for (const layer of ["eyes", "brows", "nose", "chin", "mouth"]) {
    paste(layer, cloneCanvas(masks[layer]));
  }
  return out;
}

function warpMatrix(lm) {
  const pL = lm[468];
  const pR = lm[473];
  const vx = pR[0] - pL[0];
  const vy = pR[1] - pL[1];
  const edS = Math.hypot(vx, vy);
  const ang = (Math.atan2(vy, vx) * 180) / Math.PI;
  if (edS < 60 || Math.abs(ang) >= 28) return null;
  const s = ED / edS;
  const rot = (-ang * Math.PI) / 180;
  const c = Math.cos(rot);
  const sn = Math.sin(rot);
  return [
    c * s, -sn * s, L_IRIS[0] - (c * s * pL[0] - sn * s * pL[1]),
    sn * s, c * s, L_IRIS[1] - (sn * s * pL[0] + c * s * pL[1]),
  ];
}

function boxMean(d, W, x0, y0, x1, y1) {
  x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
  x1 = Math.min(W, Math.round(x1)); y1 = Math.min(512, Math.round(y1));
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4;
      r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
    }
  }
  return n ? [r / n, g / n, b / n] : [0, 0, 0];
}

// How far the colour at the ear positions is from the cheek skin (0 = clearly visible ears).
function earSkinDistance(wd, g) {
  const cheekY0 = g.il[1] + 0.4 * ED;
  const cheekY1 = g.il[1] + 0.65 * ED;
  const sl = boxMean(wd, CANVAS, g.il[0] - 0.2 * ED, cheekY0, g.il[0] + 0.1 * ED, cheekY1);
  const sr = boxMean(wd, CANVAS, g.ir[0] - 0.1 * ED, cheekY0, g.ir[0] + 0.2 * ED, cheekY1);
  const skin = [(sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2, (sl[2] + sr[2]) / 2];
  const ey = g.eye_mid[1] + 0.25 * ED;
  const side = 0.5 * g.fw + 0.12 * ED;
  const el = boxMean(wd, CANVAS, g.face_mid[0] - side - 0.1 * ED, ey - 0.2 * ED,
                     g.face_mid[0] - side + 0.1 * ED, ey + 0.2 * ED);
  const er = boxMean(wd, CANVAS, g.face_mid[0] + side - 0.1 * ED, ey - 0.2 * ED,
                     g.face_mid[0] + side + 0.1 * ED, ey + 0.2 * ED);
  const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / 255;
  return (d3(el, skin) + d3(er, skin)) / 2;
}

function boxStats(d, W, x0, y0, x1, y1) {
  x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
  x1 = Math.min(W, Math.round(x1)); y1 = Math.min(512, Math.round(y1));
  let sum = 0, sum2 = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4;
      const g = 0.114 * d[i] + 0.587 * d[i + 1] + 0.299 * d[i + 2];
      sum += g; sum2 += g * g; n++;
    }
  }
  if (!n) return { mean: 0, std: 0 };
  const mean = sum / n;
  return { mean, std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)) };
}

// ---------------------------------------------------------------- sources

let faceLandmarker = null;
let segmenter = null;
const faceCache = new Map();
const used = new Set();
const recent = [];
let hfCooldown = 0;
let wikiPool = null;

const isNetworkError = (e) =>
  e instanceof TypeError || /Failed to fetch|Load failed|NetworkError/i.test(String(e && e.message));

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("http " + r.status);
  return r.json();
}

async function wikiBatch(n) {
  if (!wikiPool || !wikiPool.length) {
    const j = await getJSON(
      "https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers" +
      "&cmtitle=" + encodeURIComponent("Category:Portrait photographs") +
      "&cmtype=file&cmlimit=500&format=json&origin=*");
    const arr = ((j.query && j.query.categorymembers) || []).map((m) => m.title);
    shuffle(arr);
    wikiPool = arr;
  }
  const titles = wikiPool.splice(0, n);
  if (!titles.length) return [];
  const j = await getJSON(
    "https://commons.wikimedia.org/w/api.php?action=query&titles=" +
    encodeURIComponent(titles.join("|")) +
    "&prop=imageinfo&iiprop=url&iiurlwidth=1024&format=json&origin=*");
  const out = [];
  for (const p of Object.values(j.query.pages || {})) {
    const ii = p.imageinfo && p.imageinfo[0];
    const u = ii && (ii.thumburl || ii.url);
    if (u) out.push({ key: "wiki:" + p.title, url: u });
  }
  return out;
}

function loadImageTimeout(url, ms) {
  return new Promise((res, rej) => {
    const timer = setTimeout(() => rej(new Error("image timeout")), ms);
    loadImage(url).then(
      (im) => { clearTimeout(timer); res(im); },
      (e) => { clearTimeout(timer); rej(e); }
    );
  });
}

function trimCache() {
  if (faceCache.size > 30) {
    const first = faceCache.keys().next().value;
    faceCache.delete(first);
  }
}

// First inference / first canvas filter are slow (graph + shader compilation), so run the
// complete pipeline once on a small synthetic face before the first click.
async function warmUp() {
  try {
    const im = await loadImage("./models/warmup.jpg");
    const f = await analyze(im, "warmup", im.src);
    composite(planCollage(Array.from({ length: 9 }, () => ({ ...f }))));
  } catch (e) {
    console.warn("warm-up skipped:", e && e.message ? e.message : e);
  }
}

async function initModels() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm"
  );
  for (const delegate of ["GPU", "CPU"]) {
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: "./models/face_landmarker.task", delegate },
        runningMode: "IMAGE",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: "./models/selfie_segmenter.tflite", delegate },
        runningMode: "IMAGE",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      await warmUp();
      return delegate;
    } catch (e) {
      console.warn("delegate", delegate, "failed:", e);
    }
  }
  throw new Error("could not initialise MediaPipe models");
}

async function fetchRow(offset) {
  const url = "https://datasets-server.huggingface.co/rows?dataset=" +
    encodeURIComponent("PhilSad/celeba-hq-15k") +
    `&config=default&split=train&offset=${offset}&length=1`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("rows http " + r.status);
  const j = await r.json();
  const row = j.rows && j.rows[0] && j.rows[0].row;
  if (!row || !row.image || !row.image.src) throw new Error("no image in row");
  return row.image.src;
}

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("image load failed"));
    im.src = src;
  });
}

// Tiny luminance signature: catches the same picture being delivered twice
// (thispersondoesnotexist.com hands identical images to near-simultaneous requests).
const genSigs = [];

function imgSig(img) {
  const c = newCanvas(16, 16);
  const x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(img, 0, 0, 16, 16);
  const d = x.getImageData(0, 0, 16, 16).data;
  const s = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    s[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return s;
}

function registerImage(img) {
  const s = imgSig(img);
  const dup = genSigs.some((o) => {
    let t = 0;
    for (let i = 0; i < 256; i++) t += Math.abs(o[i] - s[i]);
    return t / 256 < 5;
  });
  if (dup) throw new Error("duplicate image");
  genSigs.push(s);
}

function alphaGrid(alpha) {
  const c = newCanvas(64, 64);
  const x = c.getContext("2d", { willReadFrequently: true });
  x.imageSmoothingEnabled = true;
  x.drawImage(alpha, 0, 0, 64, 64);
  const d = x.getImageData(0, 0, 64, 64).data;
  const out = new Uint8Array(4096);
  for (let i = 0; i < 4096; i++) out[i] = d[i * 4 + 3] > 127 ? 1 : 0;
  return out;
}

async function analyze(img, key, src) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const det = faceLandmarker.detect(img);
  if (!det.faceLandmarks || !det.faceLandmarks.length) throw new Error("no face");
  const lm = det.faceLandmarks[0].map((p) => [p.x * w, p.y * h]);
  const M2 = warpMatrix(lm);
  if (!M2) throw new Error("turned/too small");

  const tlm = applyM(lm, M2);
  const g = geom(tlm);
  if (Math.abs(g.yaw) > MAX_YAW) throw new Error("not front facing");

  const warped = newCanvas();
  const wx = warped.getContext("2d", { willReadFrequently: true });
  wx.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
  wx.fillRect(0, 0, CANVAS, CANVAS);
  wx.imageSmoothingEnabled = true;
  wx.imageSmoothingQuality = "high";
  setTransform(wx, M2);
  wx.drawImage(img, 0, 0);
  wx.setTransform(1, 0, 0, 1, 0, 0);

  const wd = wx.getImageData(0, 0, CANVAS, CANVAS).data;
  const eyeBox = boxStats(wd, CANVAS, EX - 0.9 * ED, EYE_Y - 0.32 * ED,
                          EX + 0.9 * ED, EYE_Y + 0.32 * ED);
  if (!(eyeBox.mean >= 78 && eyeBox.mean <= 208)) throw new Error("eye band dark/blown");
  const ls = boxStats(wd, CANVAS, L_IRIS[0] - 0.28 * ED, L_IRIS[1] - 0.16 * ED,
                      L_IRIS[0] + 0.28 * ED, L_IRIS[1] + 0.16 * ED);
  const rs = boxStats(wd, CANVAS, R_IRIS[0] - 0.28 * ED, R_IRIS[1] - 0.16 * ED,
                      R_IRIS[0] + 0.28 * ED, R_IRIS[1] + 0.16 * ED);
  if (Math.min(ls.std, rs.std) < 12 || Math.abs(ls.mean - rs.mean) > 55)
    throw new Error("glasses/lens");

  const seg = segmenter.segment(img);
  const cm = seg.categoryMask;
  const cw = cm.width;
  const ch = cm.height;
  const cat = cm.getAsUint8Array();
  const nx = clamp(Math.round((lm[1][0] / w) * cw), 0, cw - 1);
  const ny = clamp(Math.round((lm[1][1] / h) * ch), 0, ch - 1);
  const person = cat[ny * cw + nx];

  const maskSmall = newCanvas(cw, ch);
  const msx = maskSmall.getContext("2d", { willReadFrequently: true });
  const im = msx.createImageData(cw, ch);
  const md = im.data;
  for (let i = 0, p = 0; i < cat.length; i++, p += 4) {
    const on = cat[i] === person ? 255 : 0;
    md[p] = md[p + 1] = md[p + 2] = 255;
    md[p + 3] = on;
  }
  msx.putImageData(im, 0, 0);

  const maskClean = newCanvas(cw, ch);
  const mcx = maskClean.getContext("2d", { willReadFrequently: true });
  mcx.filter = "blur(4px)";
  mcx.drawImage(maskSmall, 0, 0);
  mcx.filter = "none";
  threshold(maskClean);
  const maskClean2 = newCanvas(cw, ch);
  const mc2 = maskClean2.getContext("2d", { willReadFrequently: true });
  mc2.filter = "blur(2px)";
  mc2.drawImage(maskClean, 0, 0);
  mc2.filter = "none";
  threshold(maskClean2);

  const maskSrc = newCanvas(w, h);
  const ms2 = maskSrc.getContext("2d", { willReadFrequently: true });
  ms2.imageSmoothingEnabled = true;
  ms2.imageSmoothingQuality = "high";
  ms2.drawImage(maskClean2, 0, 0, w, h);
  threshold(maskSrc);

  const alpha = newCanvas();
  const ax = alpha.getContext("2d", { willReadFrequently: true });
  setTransform(ax, M2);
  ax.imageSmoothingEnabled = true;
  ax.drawImage(maskSrc, 0, 0);
  ax.setTransform(1, 0, 0, 1, 0, 0);
  threshold(alpha);
  smoothMask(alpha);

  const ad = alpha.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, CANVAS, CANVAS).data;
  let inside = 0, total = 0;
  const x0 = Math.round(g.face_mid[0] - 0.45 * g.fw);
  const x1 = Math.round(g.face_mid[0] + 0.45 * g.fw);
  const y0 = Math.round(g.eye_mid[1] - 0.3 * g.ed);
  const y1 = Math.round(g.chin[1]);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const on = ad[(y * CANVAS + x) * 4 + 3] > 127;
      total++;
      if (on) inside++;
    }
  }
  if (!total || inside / total < 0.85) throw new Error("silhouette cut the face");

  return {
    key,
    src,
    img,
    M2,
    tlm,
    g,
    gm: mirrorGeom(g),
    warped,
    alpha,
    a64: alphaGrid(alpha),
    earDist: earSkinDistance(wd, g),
  };
}

async function getFace(offset, stop) {
  const key = "hf:" + offset;
  if (faceCache.has(key)) return faceCache.get(key);
  let src;
  try {
    src = await fetchRow(offset);
  } catch (e) {
    if (isNetworkError(e) || /http 429/.test(String(e.message)))
      hfCooldown = performance.now() + 60000;
    throw e;
  }
  const img = await loadImage(src);
  registerImage(img);
  if (stop && stop()) throw new Error("enough");
  const face = await analyze(img, key, src);
  faceCache.set(key, face);
  trimCache();
  return face;
}

async function getFaceWiki(cand, stop) {
  if (faceCache.has(cand.key)) return faceCache.get(cand.key);
  const img = await loadImageTimeout(cand.url, 12000);
  registerImage(img);
  if (stop && stop()) throw new Error("enough");
  const face = await analyze(img, cand.key, cand.url);
  faceCache.set(cand.key, face);
  trimCache();
  return face;
}

function newOffset() {
  for (let i = 0; i < 200; i++) {
    const o = Math.floor(Math.random() * 15000);
    if (used.has(o) || recent.includes("hf:" + o)) continue;
    used.add(o);
    return o;
  }
  return Math.floor(Math.random() * 15000);
}

let mode = "real";
let aiCounter = 0;

async function getFaceAI(stop) {
  const key = "ai:" + (++aiCounter) + ":" + Math.random().toString(36).slice(2);
  const src = "https://images.weserv.nl/?url=" +
    encodeURIComponent("thispersondoesnotexist.com/random-person.jpeg?cb=" +
      Math.random().toString(36).slice(2) + Date.now()) +
    "&w=1024&h=1024&fit=cover&output=jpg";
  const img = await loadImageTimeout(src, 16000);
  registerImage(img);
  if (stop && stop()) throw new Error("enough");
  // thispersondoesnotexist.com stamps "StyleGAN2 (Karras et al.)" into the bottom rows
  const c = newCanvas(img.naturalWidth, img.naturalHeight - 32);
  c.getContext("2d").drawImage(img, 0, 0);
  const face = await analyze(c, key, src);
  faceCache.set(key, face);
  trimCache();
  return face;
}

async function waitFor(tasks, enough) {
  let done = false;
  const all = Promise.allSettled(tasks).then(() => { done = true; });
  while (!done && !enough()) await Promise.race([all, sleep(50)]);
}

const POOL_TARGET = 11; // faces to choose from (9 are used)
const SOFT_MS = 10500;  // after this a pool of 9 is good enough
const HARD_MS = 18500;  // overall click budget
const AI_GAP = 320;     // spacing of AI requests (see registerImage)

async function fetchReal(need, add, enough) {
  let got = 0;
  const take = (f) => { add(f); got++; };
  if (performance.now() >= hfCooldown) {
    const offs = [];
    while (offs.length < need) offs.push(newOffset());
    let netFail = 0;
    const tasks = offs.map((o, i) =>
      sleep(i * 60).then(() => getFace(o, enough)).then(take, (e) => {
        if (isNetworkError(e)) netFail++;
      }));
    await waitFor(tasks, enough);
    if (netFail) hfCooldown = performance.now() + 60000;
  }
  if (!enough()) {
    try {
      const want = Math.max(1, need - got);
      const cands = await wikiBatch(Math.min(40, Math.max(Math.ceil(want * 2.5), 8)));
      const tasks = cands.map((c) => getFaceWiki(c, enough).then(take, () => {}));
      await waitFor(tasks, enough);
    } catch (e) { /* wiki unavailable */ }
  }
}

async function collectReal(pool, add, elapsed) {
  const enough = () => pool.length >= POOL_TARGET || (pool.length >= 9 && elapsed() > SOFT_MS);
  for (let round = 0; pool.length < 9 && elapsed() < HARD_MS && round < 6; round++) {
    await fetchReal(Math.max(POOL_TARGET - pool.length, 9 - pool.length) + 1, add, enough);
  }
}

async function collectAI(pool, add, elapsed) {
  const pending = new Set();
  const enough = () => pool.length >= POOL_TARGET || (pool.length >= 9 && elapsed() > SOFT_MS - 1500);
  while (!enough() && elapsed() < 15000) {
    if (pool.length + pending.size < POOL_TARGET + 2) {
      const p = getFaceAI(enough).then(add, () => {}).then(() => pending.delete(p));
      pending.add(p);
    }
    await sleep(AI_GAP);
  }
  while (pool.length < 9 && pending.size && elapsed() < HARD_MS) {
    await Promise.race([...pending, sleep(250)]);
  }
}

async function generate() {
  const t0 = performance.now();
  const elapsed = () => performance.now() - t0;
  genSigs.length = 0;
  const pool = [];
  const seen = new Set();
  const add = (f) => {
    if (seen.has(f.key)) return;
    seen.add(f.key);
    pool.push(f);
    recent.push(f.key);
    if (recent.length > 400) recent.shift();
  };
  if (mode === "ai") await collectAI(pool, add, elapsed);
  else await collectReal(pool, add, elapsed);
  if (pool.length < 9) throw new Error("could not collect 9 distinct faces");

  const plan = planCollage(pool);
  const canvas = composite(plan);
  const group = [plan.head, ...plan.picks.map((p) => p.face)];
  window.__plan = {
    pool: pool.length,
    earDists: pool.map((f) => +f.earDist.toFixed(2)),
    earPick: +plan.picks.find((p) => p.layer === "ears").face.earDist.toFixed(2),
    ms: Math.round(elapsed()),
    head: { key: plan.head.key, yaw: +plan.head.g.yaw.toFixed(3) },
    total: +plan.total.toFixed(2),
    picks: plan.picks.map((p) => ({
      layer: p.layer, key: p.face.key, flip: p.fit.flip,
      scale: +p.fit.sRaw.toFixed(2), rot: +(p.fit.thRaw / RAD).toFixed(1),
      cov: +p.fit.cov.toFixed(2), chk: +p.fit.chk.toFixed(2),
      yaw: +p.face.g.yaw.toFixed(2), cost: +p.fit.cost.toFixed(2),
    })),
  };
  return { canvas, group };
}

// ---------------------------------------------------------------- UI

const btn = document.getElementById("gen");
const imgEl = document.getElementById("out");
const status = document.getElementById("status");
let busy = false;
let currentURL = null;

function setStatus(html) {
  status.innerHTML = html;
}

const thumbsEl = document.getElementById("thumbs");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");
const tabButtons = Array.from(document.querySelectorAll(".tab"));
const SLOT_NAMES = ["head", ...SLOTS];

function renderThumbs(group) {
  thumbsEl.innerHTML = "";
  group.forEach((face, i) => {
    const d = document.createElement("div");
    d.className = "thumb";
    d.title = SLOT_NAMES[i] + " — " + (face.key.startsWith("ai:")
      ? "AI generated" : "source photo");
    const c = document.createElement("canvas");
    c.width = 96;
    c.height = 96;
    c.getContext("2d").drawImage(face.warped, 0, 0, 96, 96);
    const label = document.createElement("span");
    label.textContent = SLOT_NAMES[i];
    d.appendChild(c);
    d.appendChild(label);
    d.addEventListener("click", () => openLightbox(face.src));
    thumbsEl.appendChild(d);
  });
}

function openLightbox(src) {
  if (!src) return;
  lightboxImg.src = src;
  lightbox.classList.add("open");
}

function closeLightbox() {
  lightbox.classList.remove("open");
  lightboxImg.removeAttribute("src");
}

lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox || e.target === lightboxClose) closeLightbox();
});
lightboxClose.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});
tabButtons.forEach((b) => b.addEventListener("click", () => {
  if (busy) return;
  mode = b.dataset.mode;
  tabButtons.forEach((x) => x.classList.toggle("active", x === b));
  setStatus(mode === "ai"
    ? "AI faces mode: 9 fresh images from thispersondoesnotexist.com."
    : "Web mode: random portraits from CelebA-HQ / Wikimedia Commons.");
}));

btn.addEventListener("click", async () => {
  if (busy || !faceLandmarker) return;
  busy = true;
  btn.disabled = true;
  setStatus('<span class="spinner"></span> ' + (mode === "ai"
    ? "Generating 9 AI faces and cutting their features…"
    : "Searching for 9 random people and cutting their features…"));
  const t0 = performance.now();
  try {
    const { canvas, group } = await generate();
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
    const url = URL.createObjectURL(blob);
    imgEl.src = url;
    if (currentURL) URL.revokeObjectURL(currentURL);
    currentURL = url;
    renderThumbs(group);
    const ms = Math.round(performance.now() - t0);
    const aiCount = group.filter((f) => f.key.startsWith("ai:")).length;
    const hfCount = group.filter((f) => f.key.startsWith("hf:")).length;
    const wikiCount = group.length - hfCount - aiCount;
    const parts = [];
    if (aiCount) parts.push(aiCount + " AI");
    if (hfCount) parts.push(hfCount + " CelebA");
    if (wikiCount) parts.push(wikiCount + " Wikimedia");
    setStatus("New face from " + group.length + " different " +
      (aiCount === group.length ? "AI-generated people" : "people") +
      " (" + parts.join(" + ") + ") in " + ms + " ms. Click thumbnails to inspect, or Generate again.");
  } catch (e) {
    console.error(e);
    setStatus("Generation failed: " + (e && e.message ? e.message : e) + " — try again.");
  } finally {
    busy = false;
    btn.disabled = false;
  }
});

initModels().then((delegate) => {
  btn.disabled = false;
  setStatus("Models ready (" + delegate + "). Click Generate.");
}).catch((e) => {
  console.error(e);
  setStatus("Could not load models: " + (e && e.message ? e.message : e));
});
