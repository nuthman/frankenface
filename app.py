import io
import os
import glob
import random
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from flask import Flask, request, send_file, send_from_directory, Response

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "face_landmarker.task")
CACHE_DIR = os.path.join(ROOT, "cache")
DOCS_DIR = os.path.join(ROOT, "docs")
os.makedirs(CACHE_DIR, exist_ok=True)

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "PhilSad/celeba-hq-15k"
POOL_SIZE = 15000

CANVAS = 512
ED = 96           # interpupillary distance in template space (leaves room
                  # for the crown above and neck/shoulders below the face)
EYE_Y = 230       # iris row; lower than centre so the bust is not cropped
EX = 256          # face centre x
L_IRIS_T = (EX - ED // 2, EYE_Y)
R_IRIS_T = (EX + ED // 2, EYE_Y)
BG = np.array([32, 30, 28], np.float32)

app = Flask(__name__)

_tl = threading.local()
_used_lock = threading.Lock()
_used = set()
_pool = queue.Queue(maxsize=128)
_recent = deque(maxlen=100)
_breaker_until = 0.0
_breaker_fails = 0
_breaker_lock = threading.Lock()


def breaker_open():
    return time.time() < _breaker_until


def breaker_fail():
    global _breaker_until, _breaker_fails
    with _breaker_lock:
        _breaker_fails += 1
        _breaker_until = time.time() + min(60, 5 * (2 ** min(_breaker_fails, 4)))


def breaker_success():
    global _breaker_until, _breaker_fails
    with _breaker_lock:
        _breaker_until = 0.0
        _breaker_fails = 0


def get_detector():
    if not hasattr(_tl, "det"):
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _tl.det = FaceLandmarker.create_from_options(opts)
    return _tl.det


def get_session():
    if not hasattr(_tl, "sess"):
        s = requests.Session()
        s.headers["User-Agent"] = "face-frankenstein/1.0"
        _tl.sess = s
    return _tl.sess


def new_offset():
    while True:
        o = random.randrange(POOL_SIZE)
        with _used_lock:
            if o not in _used:
                _used.add(o)
                return o


def fetch_source(offset, deadline=None):
    """Return the 1024x1024 BGR image for a dataset row index, or None."""
    if breaker_open():
        return None
    for attempt in range(3):
        if deadline is not None and time.time() > deadline:
            return None
        try:
            r = get_session().get(
                HF_ROWS_URL,
                params={"dataset": DATASET, "config": "default",
                        "split": "train", "offset": offset, "length": 1},
                timeout=(6, 12),
            )
            if r.status_code != 200:
                time.sleep(1.0 * (attempt + 1))
                continue
            j = r.json()
            if "rows" not in j or not j["rows"]:
                time.sleep(1.0 * (attempt + 1))
                continue
            img_url = j["rows"][0]["row"]["image"]["src"]
            img = get_session().get(img_url, timeout=(6, 12))
            if img.status_code == 200:
                arr = np.frombuffer(img.content, np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    with open(os.path.join(CACHE_DIR, f"{offset}.jpg"), "wb") as fh:
                        fh.write(img.content)
                    breaker_success()
                    return decoded
        except (requests.RequestException, ValueError):
            time.sleep(1.0 * (attempt + 1))
    breaker_fail()
    return None


def detect_landmarks(img):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = get_detector().detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    h, w = img.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in lm], np.float32)


def person_alpha(img, offset=None, lm=None):
    """Cut the person out of the studio background with GrabCut.
    Result is cached on disk per dataset offset."""
    cache_path = None
    if offset is not None:
        cache_path = os.path.join(CACHE_DIR, f"{offset}.alpha.png")
        if os.path.exists(cache_path):
            a = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
            if a is not None and a.shape == img.shape[:2]:
                return a
    h, w = img.shape[:2]
    scale = 400.0 / max(h, w) if max(h, w) > 400 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    sh, sw = small.shape[:2]
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    b = max(2, int(0.02 * min(sh, sw)))
    mask[:b, :] = cv2.GC_BGD
    mask[-b:, :] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD
    mask[:, -b:] = cv2.GC_BGD
    fg = np.zeros((sh, sw), np.uint8)
    cv2.ellipse(fg, (sw // 2, int(0.58 * sh)),
                (int(0.34 * sw), int(0.52 * sh)), 0, 0, 360, 1, -1)
    if lm is not None:
        L = lm * (sw / float(w))
        x0, x1 = float(L[234, 0]), float(L[454, 0])
        fy0, fy1 = float(L[10, 1]), float(L[152, 1])
        cx = (x0 + x1) / 2.0
        fw = max(x1 - x0, 1.0)
        fh = max(fy1 - fy0, 1.0)
        cv2.ellipse(fg, (int(cx), int(fy0 - 0.35 * fh)),
                    (int(0.85 * fw), int(0.95 * fh)), 0, 0, 360, 1, -1)
        cv2.rectangle(fg, (int(cx - 0.45 * fw), int(fy0)),
                      (int(cx + 0.45 * fw), sh), 1, -1)
    mask[fg > 0] = cv2.GC_PR_FGD
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                     255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                             np.ones((9, 9), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    er = cv2.erode(alpha, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(er, 8)
    if n > 1:
        keep = np.where(
            labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])),
            255, 0).astype(np.uint8)
        alpha = cv2.bitwise_and(cv2.dilate(keep, k), alpha)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    alpha = np.zeros_like(alpha)
    cv2.drawContours(alpha, cnts, -1, 255, -1)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    if cache_path is not None:
        cv2.imwrite(cache_path, alpha)
    return alpha


def build_warped(img, offset):
    """Detect + sanity-check a face; return it plus its template-space map."""
    lm = detect_landmarks(img)
    if lm is None:
        return None
    pL, pR = lm[468], lm[473]
    v = pR - pL
    ed_s = float(np.linalg.norm(v))
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    if ed_s < 60 or abs(ang) >= 28:
        return None
    s = ED / ed_s
    rot = -np.radians(ang)
    c, srot = np.cos(rot), np.sin(rot)
    M2 = np.array([[c * s, -srot * s,
                   L_IRIS_T[0] - (c * s * pL[0] - srot * s * pL[1])],
                  [srot * s, c * s,
                   L_IRIS_T[1] - (srot * s * pL[0] + c * s * pL[1])]],
                 np.float32)
    M3 = np.vstack([M2, np.array([0, 0, 1], np.float32)])
    warped = cv2.warpAffine(img, M2, (CANVAS, CANVAS),
                            flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(32, 30, 28))
    alpha = person_alpha(img, offset, lm)
    alpha = cv2.warpAffine(alpha, M2, (CANVAS, CANVAS),
                           flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    eye_box = warped[int(EYE_Y - 0.32 * ED):int(EYE_Y + 0.32 * ED),
                     int(EX - 0.9 * ED):int(EX + 0.9 * ED)]
    eye_mean = float(eye_box.mean())
    if not (78.0 <= eye_mean <= 208.0):
        return None

    def eye_stats(cx, cy):
        b = warped[int(cy - 0.16 * ED):int(cy + 0.16 * ED),
                   int(cx - 0.28 * ED):int(cx + 0.28 * ED)]
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return float(g.mean()), float(g.std())

    ml, sl = eye_stats(*L_IRIS_T)
    mr, sr = eye_stats(*R_IRIS_T)
    if min(sl, sr) < 12.0 or abs(ml - mr) > 55.0:
        return None
    pts = np.hstack([lm, np.ones((lm.shape[0], 1), np.float32)])
    tlm = (M3 @ pts.T).T[:, :2].astype(np.float32)
    return {"offset": offset, "img": img, "warped": warped,
            "alpha": alpha, "M": M3, "tlm": tlm}


def make_face(offset, deadline=None):
    """Fetch, detect, and warp one source face into template space."""
    for _attempt in range(6):
        img = fetch_source(offset, deadline)
        if img is not None:
            result = build_warped(img, offset)
            if result is not None:
                return result
        if deadline is not None and time.time() > deadline:
            break
        offset = new_offset()
    # Network unusable or too slow: fall back to faces already cached on disk,
    # preferring people not used in the most recent faces.
    files = glob.glob(os.path.join(CACHE_DIR, "*.jpg"))
    random.shuffle(files)
    ids = {os.path.basename(fp)[:-4]: fp for fp in files}
    fresh = [fp for oid, fp in ids.items() if oid not in _recent]
    stale = [fp for oid, fp in ids.items() if oid in _recent]
    for fp in fresh + stale:
        img = cv2.imread(fp)
        if img is None:
            try:
                os.remove(fp)
            except OSError:
                pass
            continue
        result = build_warped(img, int(os.path.basename(fp)[:-4]))
        if result is None:
            try:
                os.remove(fp)
            except OSError:
                pass
            continue
        return result
    raise RuntimeError("no usable face (network failing and cache empty)")


def geom(tlm):
    """Named facial geometry points in template space."""
    def mid(a, b):
        return (tlm[a] + tlm[b]) / 2
    return {
        "il": tlm[468], "ir": tlm[473],
        "eye_mid": mid(468, 473),
        "ed": float(np.linalg.norm(tlm[468] - tlm[473])),
        "nose": tlm[1],
        "mouth_l": tlm[61], "mouth_r": tlm[291], "mouth_mid": mid(61, 291),
        "chin": tlm[152],
        "brow_l": tlm[23], "brow_r": tlm[300], "brow_mid": mid(23, 300),
        "face_l": tlm[234], "face_r": tlm[454], "face_mid": mid(234, 454),
        "ear_l": tlm[172], "ear_r": tlm[396],
        "top": tlm[10],
    }


def feature_specs(layer, g, grow=1.0):
    """Mask ellipses sized and positioned from a face's own geometry."""
    ed = g["ed"]
    fw = float(np.linalg.norm(g["face_r"] - g["face_l"]))
    if layer == "neck":
        specs = [(g["face_mid"][0], g["chin"][1] + 1.4 * ed,
                  0.95 * fw, 1.45 * ed)]
    elif layer == "eyes":
        specs = [(g["il"][0], g["il"][1] + 0.02 * ed, 0.30 * ed, 0.20 * ed),
                 (g["ir"][0], g["ir"][1] + 0.02 * ed, 0.30 * ed, 0.20 * ed)]
    elif layer == "brows":
        bsp = float(np.linalg.norm(g["brow_r"] - g["brow_l"]))
        specs = [(g["brow_l"][0], g["brow_l"][1], 0.36 * bsp, 0.13 * ed),
                 (g["brow_r"][0], g["brow_r"][1], 0.36 * bsp, 0.13 * ed)]
    elif layer == "nose":
        nlen = float(np.linalg.norm(g["nose"] - g["eye_mid"]))
        c = g["eye_mid"] + 0.62 * (g["nose"] - g["eye_mid"])
        specs = [(c[0], c[1], 0.44 * nlen, 0.72 * nlen)]
    elif layer == "chin":
        mw = float(np.linalg.norm(g["mouth_r"] - g["mouth_l"]))
        depth = max(g["chin"][1] - g["mouth_mid"][1], 0.3 * ed)
        specs = [(g["face_mid"][0], g["mouth_mid"][1] + 0.8 * depth,
                  0.60 * mw, 0.55 * depth)]
    elif layer == "mouth":
        mw = float(np.linalg.norm(g["mouth_r"] - g["mouth_l"]))
        specs = [(g["mouth_mid"][0], g["mouth_mid"][1], 0.72 * mw, 0.55 * mw)]
    elif layer == "ears":
        ey = g["eye_mid"][1] + 0.15 * ed
        side = 0.55 * fw
        specs = [(g["face_mid"][0] - side, ey, 0.14 * fw, 0.30 * ed),
                 (g["face_mid"][0] + side, ey, 0.14 * fw, 0.30 * ed)]
    elif layer == "hair":
        specs = [(g["face_mid"][0], g["top"][1] - 0.5 * ed,
                  0.85 * fw, 0.62 * ed)]
    else:
        raise KeyError(layer)
    return [(cx, cy, rx * grow, ry * grow) for cx, cy, rx, ry in specs]


def ellipse_mask(g, layer, grow, canvas=CANVAS):
    m = np.zeros((canvas, canvas), np.float32)
    for cx, cy, rx, ry in feature_specs(layer, g, grow):
        cv2.ellipse(m, (int(cx), int(cy)),
                    (max(int(rx), 1), max(int(ry), 1)), 0, 0, 360, 1.0, -1)
    return m


FEATURES = [
    ("neck",  "chin",      ("face_l", "face_r")),
    ("hair",  "top",       ("face_l", "face_r")),
    ("ears",  "face_mid",  ("face_l", "face_r")),
    ("eyes",  "eye_mid",   ("il", "ir")),
    ("brows", "brow_mid",  ("brow_l", "brow_r")),
    ("nose",  "nose",      ("nose", "eye_mid")),
    ("chin",  "chin",      ("chin", "mouth_mid")),
    ("mouth", "mouth_mid", ("mouth_l", "mouth_r")),
]


def region_mask(layer, g, sil):
    """Whole-silhouette regions. Hair = above the brow line inside the base
    silhouette; neck = everything below the jaw (sil=None means the donor's
    own silhouette supplies the bust shape when the base crop has no body)."""
    yy = np.arange(CANVAS, dtype=np.float32)[:, None]
    cut = g["chin"][1] - 0.25 * g["ed"]
    if layer == "hair":
        m = sil * (yy < g["eye_mid"][1] - 0.32 * g["ed"])
    elif sil is None:
        m = (yy > cut).astype(np.float32)
    else:
        m = sil * (yy > cut)
    return m.astype(np.float32)[..., None]


def blob(cx, cy, rx, ry, rng, n=18):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    j = 1.0 + np.array([rng.uniform(-0.09, 0.12) for _ in range(n)])
    x = cx + rx * j * np.cos(th)
    y = cy + ry * j * np.sin(th)
    return np.stack([x, y], 1).astype(np.int32)


def make_mask(layer, g, rng, sil=None, grow=1.0):
    m = np.zeros((CANVAS, CANVAS), np.float32)
    for cx, cy, rx, ry in feature_specs(layer, g, grow):
        cv2.fillPoly(m, [blob(cx, cy, rx, ry, rng)], 1.0)
    if sil is not None:
        m *= sil
    return m[..., None]


def slot_patch(donor, hg, layer, akey, pair):
    """Warp the donor so its feature lands where the head's feature is,
    scaled to the head's feature size. Returns patch, donor silhouette, map."""
    gd = geom(donor["tlm"])
    ad, ah = gd[akey], hg[akey]
    dm = float(np.linalg.norm(gd[pair[0]] - gd[pair[1]])) or 1.0
    hm = float(np.linalg.norm(hg[pair[0]] - hg[pair[1]]))
    r = float(np.clip(hm / dm, 0.75, 1.3))
    A = np.eye(3, dtype=np.float32)
    A[0, 0] = A[1, 1] = r
    A[0, 2] = ah[0] - r * ad[0]
    A[1, 2] = ah[1] - r * ad[1]
    M = A @ donor["M"]
    patch = cv2.warpAffine(donor["img"], M[:2], (CANVAS, CANVAS),
                           flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(32, 30, 28))
    alpha = cv2.warpAffine(donor["alpha"], A[:2], (CANVAS, CANVAS),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return patch.astype(np.float32), alpha.astype(np.float32) / 255.0, A[:2]


def warp_ellipses(g, M, layer, grow):
    """Donor feature ellipses mapped into the head's coordinate space."""
    e = ellipse_mask(g, layer, grow)
    return cv2.warpAffine(e, M, (CANVAS, CANVAS),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0.0)


def blend_cut(canvas, patch, m):
    """Pure hard paste: scissor-cut edge, no blending or halo."""
    return canvas * (1 - m) + patch * m


FACE_ERASE = {"eyes", "brows", "nose", "mouth", "chin"}


def skin_erase(canvas, m, hg, layer, sil):
    """Fill the head's own feature area (exactly the area the donor patch
    will cover) with surrounding skin tone so no ghost of the base feature
    can survive under a translucent mask edge."""
    ring = ellipse_mask(hg, layer, 1.5) - ellipse_mask(hg, layer, 1.0)
    ring[sil < 0.5] = 0.0
    v = canvas[ring > 0.5]
    if len(v) >= 300:
        canvas[m[..., 0] > 0.5] = np.median(v.reshape(-1, 3), axis=0)


CLIP_GROW = {"eyes": 1.35, "brows": 1.45, "nose": 1.6,
             "chin": 1.5, "mouth": 1.5, "ears": 1.25}


def composite(faces, seed):
    rng = random.Random(seed)
    head = faces[0]
    hg = geom(head["tlm"])
    sil = head["alpha"].astype(np.float32) / 255.0
    sil_hard = (sil > 0.5).astype(np.float32)
    canvas = np.empty((CANVAS, CANVAS, 3), np.float32)
    canvas[:, :, :] = BG
    canvas = blend_cut(canvas, head["warped"].astype(np.float32),
                       sil_hard[..., None])
    for (layer, akey, pair), donor in zip(FEATURES, faces[1:]):
        patch, d_alpha, M = slot_patch(donor, hg, layer, akey, pair)
        if layer == "hair":
            m = region_mask(layer, hg, sil) * d_alpha[..., None]
        elif layer == "neck":
            m = region_mask(layer, hg, None) * d_alpha[..., None]
        else:
            m = make_mask(layer, hg, rng, sil) * d_alpha[..., None]
            if layer in FACE_ERASE:
                gd = geom(donor["tlm"])
                clip = np.clip(
                    warp_ellipses(gd, M, layer, CLIP_GROW[layer]), 0.0, 1.0)
                if layer == "chin":
                    clip -= np.clip(warp_ellipses(gd, M, "mouth", 1.1),
                                    0.0, 1.0)
                elif layer == "eyes":
                    clip -= np.clip(warp_ellipses(gd, M, "brows", 1.15),
                                    0.0, 1.0)
                m *= clip[..., None]
        m = (m > 0.5).astype(np.float32)
        if layer in FACE_ERASE:
            skin_erase(canvas, m, hg, layer, sil)
        canvas = blend_cut(canvas, patch, m)
    return canvas


def take_faces(n=9, deadline=None):
    faces = []
    while len(faces) < n:
        try:
            faces.append(_pool.get_nowait())
        except queue.Empty:
            break
    missing = n - len(faces)
    if missing:
        offsets = [new_offset() for _ in range(missing)]
        with ThreadPoolExecutor(max_workers=missing) as ex:
            faces.extend(ex.map(lambda o: make_face(o, deadline), offsets))
    if missing:
        threading.Thread(target=_refill, args=(missing,), daemon=True).start()
    seen, distinct = set(), []

    def add(fl):
        for f in fl:
            if f["offset"] not in seen:
                seen.add(f["offset"])
                distinct.append(f)

    add(faces)
    for _round in range(3):
        if len(distinct) >= n:
            break
        extra = n - len(distinct)
        offsets = [new_offset() for _ in range(extra)]
        with ThreadPoolExecutor(max_workers=extra) as ex:
            add(list(ex.map(lambda o: make_face(o, deadline), offsets)))
    if len(distinct) < n:
        raise RuntimeError(f"only {len(distinct)} distinct people available")
    for f in distinct:
        _recent.append(str(f["offset"]))
    return distinct


def _refill(k):
    try:
        offsets = [new_offset() for _ in range(k)]
        for f in map(make_face, offsets):
            try:
                _pool.put_nowait(f)
            except queue.Full:
                break
    except RuntimeError:
        pass


def warmup():
    filled = 0
    for _attempt in range(20):
        if filled >= 12:
            break
        try:
            _pool.put_nowait(make_face(new_offset()))
            filled += 1
        except RuntimeError:
            continue


@app.route("/")
def index():
    return send_file(os.path.join(DOCS_DIR, "index.html"))


@app.route("/main.js")
def mainjs():
    return send_file(os.path.join(DOCS_DIR, "main.js"),
                     mimetype="text/javascript")


@app.route("/models/<path:name>")
def model_file(name):
    return send_from_directory(os.path.join(DOCS_DIR, "models"), name)


@app.route("/generate")
def generate():
    t0 = time.time()
    try:
        faces = take_faces(9, deadline=t0 + 15)
        canvas = composite(faces, time.time_ns())
    except RuntimeError:
        return Response("generation failed: image source unavailable", 502)
    ok, buf = cv2.imencode(".jpg", np.clip(canvas, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers["X-Persons"] = ",".join(str(f["offset"]) for f in faces)
    resp.headers["X-Elapsed-Ms"] = str(int((time.time() - t0) * 1000))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/debug/grid")
def debug_grid():
    t0 = time.time()
    try:
        faces = take_faces(9, deadline=t0 + 15)
    except RuntimeError:
        return Response("debug grid failed", 502)
    grid = []
    for f in faces:
        vis = f["warped"].copy()
        for cx, cy, rx, ry in [(256, 200, 92, 33), (256, 250, 42, 56),
                               (256, 320, 58, 40), (256, 215, 152, 178),
                               (256, 30, 205, 130)]:
            cv2.ellipse(vis, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 0), 2)
        cv2.putText(vis, f"#{f['offset']}", (10, 30), 0, 0.8, (0, 255, 255), 2)
        grid.append(vis)
    rows = [np.hstack(grid[i * 3:i * 3 + 3]) for i in range(3)]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()
    app.run(host="127.0.0.1", port=8931, threaded=True)
