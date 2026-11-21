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

const FEATURES = [
  ["neck", "chin", ["face_l", "face_r"]],
  ["hair", "top", ["face_l", "face_r"]],
  ["ears", "face_mid", ["face_l", "face_r"]],
  ["eyes", "eye_mid", ["il", "ir"]],
  ["brows", "brow_mid", ["brow_l", "brow_r"]],
  ["nose", "nose", ["nose", "eye_mid"]],
  ["chin", "chin", ["chin", "mouth_mid"]],
  ["mouth", "mouth_mid", ["mouth_l", "mouth_r"]],
];

const FACE_ERASE = new Set(["eyes", "brows", "nose", "mouth", "chin"]);
const CLIP_GROW = { eyes: 1.35, brows: 1.45, nose: 1.6, chin: 1.5, mouth: 1.5 };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function newCanvas(w = CANVAS, h = CANVAS) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
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

function geom(tlm) {
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return {
    il: tlm[468], ir: tlm[473], eye_mid: mid(tlm[468], tlm[473]),
    ed: dist(tlm[468], tlm[473]),
    nose: tlm[1], mouth_l: tlm[61], mouth_r: tlm[291],
    mouth_mid: mid(tlm[61], tlm[291]), chin: tlm[152],
    brow_l: tlm[23], brow_r: tlm[300], brow_mid: mid(tlm[23], tlm[300]),
    face_l: tlm[234], face_r: tlm[454], face_mid: mid(tlm[234], tlm[454]),
    top: tlm[10],
  };
}

function specs(layer, g, grow = 1) {
  const ed = g.ed;
  const fw = dist(g.face_r, g.face_l);
  const P = (x, y, rx, ry) => ({ x, y, rx: rx * grow, ry: ry * grow });
  switch (layer) {
    case "neck":
      return [P(g.face_mid[0], g.chin[1] + 1.4 * ed, 0.95 * fw, 1.45 * ed)];
    case "ears": {
      const ey = g.eye_mid[1] + 0.15 * ed;
      const side = 0.55 * fw;
      return [P(g.face_mid[0] - side, ey, 0.14 * fw, 0.3 * ed),
              P(g.face_mid[0] + side, ey, 0.14 * fw, 0.3 * ed)];
    }
    case "eyes":
      return [P(g.il[0], g.il[1] + 0.02 * ed, 0.3 * ed, 0.2 * ed),
              P(g.ir[0], g.ir[1] + 0.02 * ed, 0.3 * ed, 0.2 * ed)];
    case "brows": {
      const sp = dist(g.brow_r, g.brow_l);
      return [P(g.brow_l[0], g.brow_l[1], 0.36 * sp, 0.13 * ed),
              P(g.brow_r[0], g.brow_r[1], 0.36 * sp, 0.13 * ed)];
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
      const qx = A.r * px + A.tx;
      py = A.r * py + A.ty;
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

function destIn(c, other) {
  const x = c.getContext("2d", { willReadFrequently: true });
  x.globalCompositeOperation = "destination-in";
  x.drawImage(other, 0, 0);
  x.globalCompositeOperation = "source-over";
  return threshold(c);
}

function featureMask(layer, g, sil) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  for (const s of specs(layer, g)) blobPath(x, s.x, s.y, s.rx, s.ry);
  threshold(c);
  if (sil) destIn(c, sil);
  return threshold(c);
}

function regionMask(layer, g, sil) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  if (layer === "hair") {
    const y = g.eye_mid[1] - 0.32 * g.ed;
    x.fillRect(0, 0, CANVAS, Math.max(0, y));
  } else {
    const top = g.chin[1] - 0.25 * g.ed;
    const fw = dist(g.face_r, g.face_l);
    const cx = g.face_mid[0];
    x.beginPath();
    x.moveTo(cx - 0.55 * fw, top);
    x.lineTo(cx + 0.55 * fw, top);
    x.lineTo(cx + 0.95 * fw, CANVAS);
    x.lineTo(cx - 0.95 * fw, CANVAS);
    x.closePath();
    x.fill();
  }
  threshold(c);
  if (sil) destIn(c, sil);
  return threshold(c);
}

function clipCanvas(donorGeom, A, layer, subtract) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.fillStyle = "#fff";
  for (const s of specs(layer, donorGeom, CLIP_GROW[layer]))
    ellipsePath(x, s.x, s.y, s.rx, s.ry, A);
  threshold(c);
  if (subtract) {
    const sub = newCanvas();
    const sx = sub.getContext("2d");
    sx.fillStyle = "#fff";
    for (const s of specs(subtract.layer, donorGeom, subtract.grow))
      ellipsePath(sx, s.x, s.y, s.rx, s.ry, A);
    threshold(sub);
    const x2 = c.getContext("2d", { willReadFrequently: true });
    x2.globalCompositeOperation = "destination-out";
    x2.drawImage(sub, 0, 0);
    x2.globalCompositeOperation = "source-over";
    threshold(c);
  }
  return threshold(c);
}

function warpAlpha(alphaCanvas, A) {
  const c = newCanvas();
  const x = c.getContext("2d", { willReadFrequently: true });
  x.setTransform(A.r, 0, 0, A.r, A.tx, A.ty);
  x.drawImage(alphaCanvas, 0, 0);
  x.setTransform(1, 0, 0, 1, 0, 0);
  return threshold(c);
}

function patchCanvas(donor, hg, akey, pair) {
  const gd = geom(donor.tlm);
  const dm = dist(gd[pair[0]], gd[pair[1]]) || 1;
  const hm = dist(hg[pair[0]], hg[pair[1]]);
  const r = clamp(hm / dm, 0.75, 1.3);
  const A = { r, tx: hg[akey][0] - r * gd[akey][0], ty: hg[akey][1] - r * gd[akey][1] };
  const c = newCanvas();
  const x = c.getContext("2d");
  x.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
  x.fillRect(0, 0, CANVAS, CANVAS);
  x.imageSmoothingEnabled = true;
  x.imageSmoothingQuality = "high";
  x.setTransform(r, 0, 0, r, A.tx, A.ty);
  x.drawImage(donor.warped, 0, 0);
  x.setTransform(1, 0, 0, 1, 0, 0);
  return { canvas: c, A };
}

function composite(faces) {
  const out = newCanvas();
  const ox = out.getContext("2d");
  ox.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
  ox.fillRect(0, 0, CANVAS, CANVAS);

  const head = faces[0];
  const hg = geom(head.tlm);
  const sil = head.alpha;
  const hp = newCanvas();
  hp.getContext("2d").drawImage(head.warped, 0, 0);
  destIn(hp, sil);
  ox.drawImage(hp, 0, 0);

  FEATURES.forEach(([layer, akey, pair], i) => {
    const donor = faces[i + 1];
    const { canvas: patch, A } = patchCanvas(donor, hg, akey, pair);
    let m;
    if (layer === "hair") m = regionMask(layer, hg, sil);
    else if (layer === "neck") m = regionMask(layer, hg, null);
    else m = featureMask(layer, hg, sil);
    if (FACE_ERASE.has(layer)) {
      const gd = geom(donor.tlm);
      const subtract =
        layer === "chin" ? { layer: "mouth", grow: 1.1 }
        : layer === "eyes" ? { layer: "brows", grow: 1.15 }
        : null;
      m = destIn(m, clipCanvas(gd, A, layer, subtract));
    }
    m = destIn(m, warpAlpha(donor.alpha, A));
    m = smoothMask(m);
    destIn(patch, m);
    ox.drawImage(patch, 0, 0);
  });
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
    for (let i = arr.length - 1; i > 0; i--) {
      const k = (Math.random() * (i + 1)) | 0;
      [arr[i], arr[k]] = [arr[k], arr[i]];
    }
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

async function analyze(img, key) {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const det = faceLandmarker.detect(img);
  if (!det.faceLandmarks || !det.faceLandmarks.length) throw new Error("no face");
  const lm = det.faceLandmarks[0].map((p) => [p.x * w, p.y * h]);
  const M2 = warpMatrix(lm);
  if (!M2) throw new Error("turned/too small");

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

  const tlm = applyM(lm, M2);
  const g = geom(tlm);
  const fw = dist(g.face_r, g.face_l);
  const ad = alpha.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, CANVAS, CANVAS).data;
  let inside = 0, total = 0;
  const x0 = Math.round(g.face_mid[0] - 0.45 * fw);
  const x1 = Math.round(g.face_mid[0] + 0.45 * fw);
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
    img,
    M2,
    tlm,
    warped,
    alpha,
  };
}

async function getFace(offset) {
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
  const face = await analyze(img, key);
  faceCache.set(key, face);
  trimCache();
  return face;
}

async function getFaceWiki(cand) {
  if (faceCache.has(cand.key)) return faceCache.get(cand.key);
  const img = await loadImageTimeout(cand.url, 12000);
  const face = await analyze(img, cand.key);
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

async function generate() {
  const t0 = performance.now();
  const faces = [];
  const seen = new Set();
  const failures = [];
  const timeLeft = () => performance.now() - t0 < 18500;
  const push = (f) => {
    if (!seen.has(f.key)) {
      seen.add(f.key);
      faces.push(f);
      recent.push(f.key);
      if (recent.length > 400) recent.shift();
    }
  };

  while (faces.length < 9 && timeLeft() && performance.now() >= hfCooldown) {
    const need = 9 - faces.length;
    const offs = [];
    while (offs.length < need) offs.push(newOffset());
    const results = await Promise.allSettled(offs.map((o) => getFace(o)));
    let netFail = 0;
    for (const r of results) {
      if (r.status === "fulfilled") push(r.value);
      else {
        failures.push(r.reason && r.reason.message);
        if (isNetworkError(r.reason)) netFail++;
      }
    }
    if (netFail) {
      hfCooldown = performance.now() + 60000;
      break;
    }
  }

  while (faces.length < 9 && timeLeft()) {
    const need = 9 - faces.length;
    let cands = [];
    try {
      cands = await wikiBatch(Math.min(50, Math.max(need * 3, 9)));
    } catch (e) {
      failures.push(e && e.message);
      break;
    }
    if (!cands.length) break;
    const results = await Promise.allSettled(cands.map((c) => getFaceWiki(c)));
    for (const r of results) {
      if (r.status === "fulfilled") push(r.value);
      else failures.push(r.reason && r.reason.message);
    }
  }

  if (faces.length < 9) {
    throw new Error("could not collect 9 distinct faces (" +
      failures.slice(0, 3).join(", ") + ")");
  }
  const group = faces.slice(0, 9);
  return { canvas: composite(group), offsets: group.map((f) => f.key) };
}

const btn = document.getElementById("gen");
const imgEl = document.getElementById("out");
const status = document.getElementById("status");
let busy = false;
let currentURL = null;

function setStatus(html) {
  status.innerHTML = html;
}

btn.addEventListener("click", async () => {
  if (busy || !faceLandmarker) return;
  busy = true;
  btn.disabled = true;
  setStatus('<span class="spinner"></span> Searching for 9 random people and cutting their features…');
  const t0 = performance.now();
  try {
    const { canvas, offsets } = await generate();
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
    const url = URL.createObjectURL(blob);
    imgEl.src = url;
    if (currentURL) URL.revokeObjectURL(currentURL);
    currentURL = url;
    const ms = Math.round(performance.now() - t0);
    const hfCount = offsets.filter((o) => o.startsWith("hf:")).length;
    const wikiCount = offsets.length - hfCount;
    const src = hfCount + " CelebA" + (wikiCount ? " + " + wikiCount + " Wikimedia" : "");
    setStatus("New face from " + offsets.length + " different people (" + src +
      ") in " + ms + " ms. Click Generate again.");
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
