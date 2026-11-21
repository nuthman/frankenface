# Face Frankenstein — progress

## Requirements
- Webpage: "Generate" button at top, image display fills the rest of the screen.
- Each click: get random front-facing photos of random people from the internet,
  cut one feature from each of 9 different people (eyes, nose, mouth, ears, chin,
  eyebrows, head, hair, neck+shoulders) and composite them into ONE face.
- Every click produces a new face; generation must finish in < 20 s.

## Everything implemented (all verified by running it)
- `app.py` — Flask server: `/` page, `/generate` JPEG, `/debug/grid` alignment montage.
- `static/index.html` — Generate button header, `<img>` stage filling the rest,
  spinner while generating, per-click status line showing the 9 source person ids + ms.
- Face pipeline: MediaPipe Tasks FaceLandmarker (478 landmarks) → similarity-transform
  alignment on the two iris centers → GrabCut person cutout → per-feature pieces
  aligned to the head's landmarks, clipped to the head's silhouette and the donor's
  own feature/silhouette → hard-edged collage paste + color harmonization.
- Quality filters: reject face if no landmarks, tiny/rotated face, sunglasses (dark
  eye-region mean), or blown-out eyes.
- Robustness: HF rate-limit/429 handling, 15 s per-click deadline, circuit breaker
  that stops hammering the source for up to 60 s after failures, and a disk cache
  (`cache/`, auto-filled from every download) used as instant fallback while the
  source is throttled. People used in recent faces are avoided when falling back.
- Speed measured: warm clicks 0.15–0.2 s, cold click 1.2 s, throttled clicks 0.2 s,
  recovery click 2.4 s, hard cap 15 s (composite itself ~0.05 s).

## Decisions made
- **Image source**: `PhilSad/celeba-hq-15k` on Hugging Face via the public
  datasets-server `/rows` endpoint (random offset 0..14999 → 1024×1024 real
  front-facing portrait photos, no API key).
  - `thispersondoesnotexist.com` returns nginx 404 today (verified), and
    `this-person-does-not-exist.com` serves article/JS-challenge HTML, so both
    are unusable.
  - The 70k FFHQ mirror (`Ryan-sjtu/ffhq512-caption`) is rejected by datasets-server
    with a parquet row-group scan-size limit, so the 15k CelebA-HQ mirror is used.
- **Alignment/compositing** is server-side Python (so results could be visually
  inspected with the Read tool during testing); 512×512 template, ED=112 px.
- **Distinct people**: process-wide `used` offset set; per-batch uniqueness enforced
  after assembly (verified: 0 duplicate ids within any click's 9 ids).
- **Feature→person assignment**: each click assigns 9 fresh random people to the
  9 slots (head, neck, eyes, brows, nose, chin, mouth, ears, hair).

## Test log
- Landmark detection on sample face: 478 landmarks, iris/nose/mouth/chin points sane.
- Debug grid: confirmed alignment after fixing a rotation-sign bug.
- Composites out1..out_cold/qrec visually inspected: mosaic faces confirmed; fixed
  sunglasses bleed-through, brow patches covering eyes, ear patches grabbing
  background, and skin-tone mismatches (harmonization).
- Stress: 6 rapid clicks all 200 OK at 0.15–0.2 s with breaker active; after breaker
  expiry a live click re-downloaded 7 new sources (cache grew 61→68) in 2.4 s.
- Worst case seen: 19.6 s once when HF throttled hard (before deadline/breaker existed)
  — fixed by the 15 s deadline + cache fallback; no click now exceeds ~16 s.
- Frontend inline JS: syntax-checked with node; `GET /` returns the page (200).

## Caveats / next ideas
- A few source photos carry Getty-style watermarks baked into the image; unavoidable
  with this mirror (small patches sometimes make them visible).
- Side-lit faces can leave a light/shadow mismatch inside a single patch; only
  global per-patch color harmonization is implemented (per-region transfer would
  polish this further).
- Could later swap in a Flickr/Unsplash key-based source for unlimited real photos.

## Update 2 (requested 2026-11-21): collage-style rework
User asked for: better feature scaling, no double features (bleed-through), and a
magazine cut-out look instead of feathered blending.

### What changed
- **Head-anchored geometry**: each face now carries its 478 landmarks in template
  space (`tlm`). The head person defines the target geometry; every other feature
  layer gets an extra similarity transform (`slot_patch`) that moves the donor's
  feature landmark (brow mid, nose tip, mouth mid, chin, face center, forehead
  top) exactly onto the head's corresponding landmark and rescales the donor
  feature to the head's feature size (ratio clamped 0.75–1.3). This fixes both
  "scaled a bit better" and misplacement.
- **No bleed-through**: masks are now derived from the *head person's* geometry
  (eyes at head eye line sized by head ED, mouth sized by head mouth width, etc.)
  and are ~1.2–1.4× the head's own feature extent, so the base head's underlying
  feature is always fully replaced within the mask — no double noses/chins.
- **Cut-out look**: masks are jittered radial polygons (±6–10% per vertex),
  nearly hard edges (σ=1.6 blur only for anti-aliasing), plus a thin bright "paper
  edge" ring (`blend_cut`) at every patch border → magazine collage feel.
- **Harmonization** now samples statistics only inside the final mask region
  (fixes skin-tone islands), with strength clamped to a=0.8–1.3, shift ±60.
- **Hair patch** is capped above the brows so the hair donor's own eyes/forehead
  don't intrude.

### Rejected approach (documented)
- A "feature window clutter" filter (ring-vs-center color distance > 60 at eye/nose/
  mouth anchors) was tried to reject faces with hands/clothing over features, but
  measurement showed normal mouths/eyes naturally score 60–130 → it rejected ~97%
  of candidates (duplicate people within batches, 17 s clicks). Removed; occasional
  background chunks in patches are accepted as part of the collage aesthetic.

### Verified after rework
- Standalone composites (new5..new9) + live clicks (x1..x3): readable collage faces,
  cut edges visible, no double features.
- 3 live clicks: 0 duplicate offsets within any batch (take_faces retry rounds +
  prune-invalid-cache); timings 0.12 s / 6.4 s / 0.14 s.

## Update 3 (requested 2026-11-21): base-face silhouette + true head-shaped cutout
User reported: double features still visible (esp. double noses), hair still a blob
floating on top, and proposed: cut out the whole base face+shoulders as the base
layer, then clip every donor piece by the base face's outline so nothing can display
outside it.

### What changed (app.py)
- **Base silhouette layer** (`person_alpha`): every source photo is cut out of its
  studio background with GrabCut (480 px, 5 iters, probable-foreground priors from
  the face landmarks: head cone above the forehead + neck column + bust ellipse),
  post-processed (close, largest component, external-contour hole fill) and cached
  per offset as `cache/<offset>.alpha.png`. The head person's full person alpha
  (hair, jawline, neck, shoulders) is pasted as the base layer, so the final
  composite is exactly that head-shaped/collage silhouette on the flat background.
- **All pieces clipped to the base outline**: every donor patch mask is multiplied
  by the base silhouette (`sil`), so no piece can float beyond the base head.
- **No double features**: per-slot masks still come from the head's geometry; in
  addition each donor patch is clipped to the donor's *own* feature ellipse (warped
  into head space, per-layer growth 1.25–1.6) so a donor's mouth/eyes can never ride
  along inside another slot's patch (this was the source of the remaining double
  nose / double mouth). Donor patches are also clipped to the donor's own silhouette
  (`warpAffine(alpha, A)`), removing background/glare chunks.
- **Guaranteed replacement**: under each face-feature mask (`eyes/brows/nose/mouth/
  chin`) the base feature is first filled with the surrounding skin tone sampled
  from a ring around the patch (`skin_erase`) before the donor piece is pasted, so
  no ghost of the base nose/eye can survive under a soft mask edge.
- **Hair and neck are region transplants, not blobs**: hair = everything inside the
  base silhouette above the brow line; neck = everything inside the silhouette below
  the jaw. Both use the donor pixel content clipped by the donor's own silhouette,
  so the base person's hair remains visible behind/around the donor hair (layered
  hairline, exactly the "more red hair behind the blonde" effect requested) and the
  composite keeps the base person's head/jaw outline.
- **Eyes split per eye** (two iris-centred pieces) instead of one wide band; the
  donor's brows are subtracted from the eye clip so brows can only come from the
  brows slot (no double eyebrows). Ears moved to the head's sides, pasted before
  the face features.
- **Quality filter tightened** for the frontal-collage assumption: iris rotation
  < 28°, eye-region mean brightness 78–208 (rejects sunglasses and blown eyes).
- Paper-edge highlight softened (weight 0.22, σ=2.0).

### Verified after update 3
- Silhouette montage (`debug/alphas.jpg`): hair and hats are included in the base
  cutout on all six sampled faces; magazine text/gradient backgrounds removed.
- Standalone composites on 27 cached faces (`debug/m1..m6.jpg`) and live clicks
  (`debug/final1..final4.jpg`): head-shaped composites, hair inside the head
  outline, single nose/mouth/eye pair, no pieces beyond the base border.
- Live `/generate` x4 after restart: 2.8–4.3 s each (GrabCut alpha is cached per
  person after first use), 9 distinct offsets per click, HTTP 200.

## Update 4 (requested 2026-11-21): hair/ears/neck were missing, one eye black
User screenshot: correct head outline, but no hair, ears, neck or shoulders, and
one eye was a black hole.

### Causes and fixes
- **Black "missing" eye**: the eye donor was wearing dark glasses (one lens nearly
  black). Added a glasses/lens rejection in `build_warped`: per-eye windows around
  the iris landmarks, reject if `min(std) < 12` (uniform lens) or
  `|mean_left - mean_right| > 55` (one dark lens). Calibrated on 40 cached faces
  (normal eyes have std >= ~25).
- **Washed-out / missing hair**: (a) the crown was cropped by the template scale and
  (b) harmonization was recoloring dark hair toward the head person's skin tone.
  Fixes: template retuned (`ED` 112 -> 96, `EYE_Y` 200 -> 230) so the whole source
  portrait fits (crown above, bust below); hair and neck pieces skip harmonization
  so donor hair/clothing colors survive.
- **No ears**: ear pieces moved from the old cheek landmarks to the head sides
  (+-0.55 face width, at eye+0.15ED), slightly larger (0.14fw x 0.30ED) and no
  longer clipped by the donor's own tiny ear ellipse, so real ear-region content
  lands at the head edges (visible in the live composites).
- **No neck/shoulders**: the base person's crop often stops at the jaw, so the
  silhouette-clipped neck had nothing to show. The neck piece is now driven by the
  *donor's* silhouette below a line 0.25 ED above the head chin (`region_mask`
  with sil=None), so the donor's neck/chest supplies the bust and connects to the
  jaw. Chin patch shrunk (0.60 MW x 0.55 depth) so it no longer reads as a flat
  beard covering the neck.
  - Dataset caveat: CelebA-HQ crops put the chin at ~80% of frame height, so at
    most ~19% of the frame is neck/chest. The composite shows a neck + upper-chest
    strip; full shoulders would need a bust-framed source.
- **Speed**: GrabCut at 400 px / 4 iterations (was 480/5) to shorten cold clicks.

### Verified after update 4
- Standalone re-tests `debug/n1..n6.jpg`, `debug/p1..p6.jpg`, `debug/q1..q6.jpg`,
  `debug/s1..s3.jpg`: hair keeps donor color (dark/red/brown), crown visible, ear
  pieces present, neck piece connected to the jaw.
- Live `/generate` x4 after restart: 1.8 s, 1.9 s, 2.4 s, 6.6 s; 9 distinct offsets
  each; both eyes present in every composite; HTTP 200.

## Update 5 (requested 2026-11-21): true scissor-cut collage, colors untouched
User reference: a classic scanned paper collage with pieces in completely different
color temperatures and lighting, edges cut hard with scissors. No harmonization,
no soft edges.

### What changed
- **Harmonization removed entirely**: deleted `harmonize`/`region_stats`; donor
  pieces keep their original colors, white balance, and lighting (reference example
  has a green-gray cheek next to a warm one).
- **Binary scissor masks**: every mask (feature blobs, region transplants, donor
  feature clips, donor silhouette, base silhouette) is thresholded to 0/1 and
  pasted with `canvas*(1-m) + patch*m` — no anti-aliased blend, no halo. Removed
  the old Gaussian blur on masks and the paper-edge highlight ring in `blend_cut`.
- **No content smoothing**: patches are no longer Gaussian-blurred before pasting.
- **More angular pieces**: blob polygons reduced from 44 to 18 vertices with
  ±9–12% per-vertex jitter → visibly scissor-cut, hand-cut polygon edges instead of
  smooth wavy contours.
- Composite still clips pieces to the base silhouette (Update 3), now with a hard
  thresholded silhouette edge.

### Verified after update 5
- Standalone `debug/u1,u2.jpg` and live `debug/v1..v3.jpg`: hard edges everywhere,
  adjacent pieces clearly different colors/lighting, no halos or feathering.
- Live `/generate` x3 after restart: 47 ms (warm pool), 3.2 s, 5.5 s; 9 distinct
  offsets each; HTTP 200.

## Update 6 (2026-11-21): static GitHub Pages build (browser-only pipeline)
GitHub Pages cannot run Flask, so a fully client-side build was added in `docs/`
(published as the Pages site). The Flask app remains the local/server version.

- `docs/index.html` + `docs/main.js`: the whole pipeline runs in the browser —
  MediaPipe Tasks Vision JS (`FaceLandmarker` + `ImageSegmenter` selfie model),
  canvas compositing, hard binary scissor masks, no harmonization (same rules as
  Update 5). Models are bundled same-origin in `docs/models/` so no CORS issues:
  `face_landmarker.task`, `selfie_segmenter.tflite`.
- Person silhouette: `ImageSegmenter` (selfie model) replaces OpenCV GrabCut;
  category class chosen from the nose-tip pixel, thresholded, warped to the 512
  template.
- Same source: HF `datasets-server` rows + cached-assets images (CORS verified
  `access-control-allow-origin: *`), random offsets 0..14999, in-memory face cache,
  per-click 9 distinct offsets, 18.5 s collection deadline.
- Neck piece constrained to a bust trapezoid (0.55 face-width at the jaw widening
  to 0.95 at the frame bottom) so it no longer spans the full canvas width.
- Tested headless Chrome (puppeteer-core + installed Chrome): models load on GPU,
  first click generates a face from 9 distinct people in ~11 s, hard-cut collage
  output confirmed visually (`debug/web_out1` screenshots in earlier runs).

### Update 6.1: clean scissor edges (fragmented edges fixed)
The selfie segmenter mask is 256x256; nearest upscaling + thresholding produced
staircase/fragmented edges (visible in a user zoom). Fix in `docs/main.js`:
- smooth the category mask at native 256 res with two blur+threshold passes
  (4 px then 2 px) before scaling up (bilinear, high quality) and warping;
- a final `smoothMask` (blur 1.4 px + threshold) on every completed patch mask
  removes 1-px specks/islands while keeping a hard edge;
- new face-coverage check: if the person mask covers less than 85% of the face
  centre box the face is rejected (`silhouette cut the face`) and another random
  person is fetched, preventing half-face results from bad segmentations.
Verified headless: zoomed crop shows a clean continuous hairline cut with no
fragmentation; full face still composites from 9 distinct people in ~11.5 s.

### Update 6.2: image-source fallback (HF rate limiting broke generation)
HF `datasets-server` started returning 429 from CloudFront without CORS headers,
which browsers surface as `TypeError: Failed to fetch` — every click failed with
"could not collect 9 distinct faces". Fix in `docs/main.js`:
- primary remains HF rows/cached-assets;
- on a network-type failure or HTTP 429 the client sets a 60 s HF cooldown and
  transparently falls back to Wikimedia Commons (`Category:Portrait photographs`,
  `categorymembers` + `imageinfo` thumbnails at 1024 px, CORS verified). Candidates
  are shuffled, used once each, and the normal face checks (frontal, eyes, glasses,
  silhouette coverage) still apply; measured 55% valid faces on a 40-image sample.
- images keep loading in parallel; per-click deadline raised to 18.5 s.
Verified headless with HF requests deliberately aborted: "9 different people
(0 CelebA + 9 Wikimedia) in 13.8 s", clean hard-cut collage output.

## How to run
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py        # serves http://127.0.0.1:8931
```
Hosted static version (browser-only): https://nuthman.github.io/face-frankenstein/
(source in `docs/`, served by GitHub Pages). Repo: nuthman/face-frankenstein.
