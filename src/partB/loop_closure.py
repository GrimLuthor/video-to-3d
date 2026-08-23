"""Loop-closure matching for the incremental-SfM front-end (Part B; ADDITIVE to
Part A, edits nothing).

Why this is needed
------------------
Part A's `feature_tracks.match_windowed` only links frame i to frames i+1..i+W.
That is exactly right for a one-way walk, but it is BLIND to a trajectory that
RETURNS to an earlier place: an orbit around an object, or a building loop. The
frames that close the loop (the end of the ring viewing the same surface as its
start) are hundreds of frames apart, far outside the window, so no correspondence
ever ties them together. The reconstruction is then a chain wrapped into a ring
with no constraint forcing the two ends to meet: scale/pose drift accumulates
once around and the loop does not close.

What this module adds
---------------------
The classic place-recognition + geometric-verification loop-closure front-end:

  1. Build a small bag-of-visual-words vocabulary from THIS video's own RootSIFT
     descriptors (cv2.kmeans), and represent every frame as an idf-weighted,
     L2-normalised word histogram (a global image descriptor).
  2. For each frame, retrieve its most similar TEMPORALLY-DISTANT frames
     (|i-j| >= min_gap, i.e. outside the windowed matcher's reach) by cosine
     similarity of those BoW vectors -- the cheap candidate step.
  3. Geometrically VERIFY each candidate pair with the SAME mutual-ratio match +
     essential-matrix RANSAC Part A uses (reusing feature_tracks._match_pair and
     its MAGSAC estimator), keeping only pairs with a strong inlier set.

The surviving `(i, j, idx_i, idx_j)` tuples are in EXACTLY the format
`feature_tracks.build_tracks` consumes, so appending them to the windowed matches
makes a physical point seen BEFORE and AFTER the loop a SINGLE track. That track
spans a very wide baseline (the whole loop), so incremental_sfm triangulates it
across the ring and the global bundle adjustment must satisfy its reprojection in
both the early and late cameras -- which pins the loop closed and makes the whole
reconstruction globally scale-consistent. No pose-graph step is required: global
BA over the shared loop observations IS the loop closure.

Verification is deliberately STRICTER than the windowed matcher (higher inlier
floor): a false loop closure fuses two unrelated places and corrupts the map,
whereas a missed one only forgoes a constraint. build_tracks additionally
discards any track that ends up observing one frame twice, and incremental_sfm's
pose-jump guard rejects a wildly displaced registration -- two more safety nets.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1]  # Final/src
sys.path.insert(0, str(SRC))
# Reuse Part A's pairwise matcher + essential-matrix estimator read-only, so loop
# closures are verified by identical geometry to the windowed matches.
from feature_tracks import _match_pair, _E_METHOD  # noqa: E402


def build_vocabulary(features, n_words=256, sample_cap=40000, seed=0, verbose=True):
    """K-means visual vocabulary over a sample of all frames' RootSIFT descriptors.

    Returns the (n_words, 128) float32 cluster centres (fewer if there are very
    few descriptors). Falls back gracefully when the video is tiny."""
    descs = [f["desc"] for f in features if f["desc"] is not None and len(f["desc"])]
    if not descs:
        return None
    alld = np.concatenate(descs, axis=0).astype(np.float32)
    if len(alld) > sample_cap:
        rng = np.random.default_rng(seed)
        alld = alld[rng.choice(len(alld), sample_cap, replace=False)]
    k = int(min(n_words, max(2, len(alld) // 40)))  # keep clusters well-populated
    if len(alld) < k or k < 2:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(alld, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    if verbose:
        print(f"  BoW vocabulary: {k} visual words from {len(alld)} sampled descriptors")
    return centers.astype(np.float32)


def _assign_words(descs, vocab):
    """Nearest visual word for each descriptor (FLANN KD-tree over the vocab)."""
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    out = []
    for d in descs:
        if d is None or len(d) == 0:
            out.append(np.zeros(0, int))
            continue
        m = matcher.knnMatch(np.asarray(d, np.float32), vocab, k=1)
        out.append(np.array([mm[0].trainIdx for mm in m if mm], int))
    return out


def bow_vectors(features, vocab):
    """idf-weighted, L2-normalised bag-of-words histogram per frame (N, n_words)."""
    n_words = len(vocab)
    words = _assign_words([f["desc"] for f in features], vocab)
    tf = np.zeros((len(features), n_words), np.float32)
    for i, w in enumerate(words):
        if len(w):
            np.add.at(tf[i], w, 1.0)
    df = (tf > 0).sum(axis=0)                      # frames containing each word
    idf = np.log((len(features) + 1.0) / (df + 1.0)) + 1.0
    vecs = tf * idf[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _mean_descriptor_vectors(features):
    """Fallback global descriptor when a vocabulary can't be built: L2-normalised
    mean RootSIFT descriptor per frame. Weaker than BoW but fine as a coarse
    candidate filter (geometric verification is the real gate)."""
    vecs = np.zeros((len(features), 128), np.float32)
    for i, f in enumerate(features):
        if f["desc"] is not None and len(f["desc"]):
            vecs[i] = f["desc"].mean(axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def retrieve_candidates(bow, min_gap, top_k=6, min_sim=0.0):
    """For each frame, its top-k most similar TEMPORALLY-DISTANT frames.

    Returns a sorted list of unique (i, j) pairs with i < j and j - i >= min_gap,
    ranked (descending) by BoW cosine similarity. NOTE: a global appearance
    descriptor is not always discriminative enough to rank a true revisit first
    (on an object orbit every frame shares background/texture, so similarities are
    near-flat and the genuine loop pair can be out-ranked). BoW retrieval catches
    mid-sequence revisits; the sequential WRAP candidates below are the reliable
    workhorse for a trajectory that returns near where it started."""
    n = len(bow)
    sim = bow @ bow.T
    cand = {}
    for i in range(n):
        s = sim[i].copy()
        far = np.abs(np.arange(n) - i) >= min_gap   # skip window-covered neighbours
        s[~far] = -np.inf
        order = np.argsort(s)[::-1][:top_k]
        for j in order:
            if not np.isfinite(s[j]) or s[j] < min_sim:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                cand[(a, b)] = max(cand.get((a, b), -1.0), float(s[j]))
    return [p for p, _ in sorted(cand.items(), key=lambda kv: -kv[1])]


def wrap_candidates(n, span, min_gap):
    """Sequential loop-closure candidates: every (head-block frame, tail-block
    frame) pair. A capture that ORBITS an object or walks a loop ends near where
    it began, so the closure is between the first `span` frames and the last
    `span` frames -- and unlike appearance retrieval this needs no descriptor to
    be discriminative, only that the ends geometrically overlap (verified next)."""
    span = int(max(1, min(span, n // 2)))
    pairs = []
    for i in range(span):
        for j in range(n - span, n):
            if j - i >= min_gap:
                pairs.append((i, j))
    return pairs


def verify_pairs(features, K, candidates, ratio_thresh=0.75, ransac_thresh=1.0,
                 min_inliers=30, max_pairs=None, verbose=True):
    """Match + essential-matrix RANSAC each candidate; keep strong inlier pairs.

    Same geometry as feature_tracks.match_windowed, but a higher inlier floor: a
    false loop closure is far more damaging than a missed one. Returns tuples
    (i, j, idx_i_inliers, idx_j_inliers) ready for feature_tracks.build_tracks."""
    verified, gaps = [], []
    for n_done, (i, j) in enumerate(candidates):
        if max_pairs is not None and len(verified) >= max_pairs:
            break
        idx_a, idx_b = _match_pair(features[i]["desc"], features[j]["desc"], ratio_thresh)
        if len(idx_a) < min_inliers:
            continue
        pts_a = features[i]["xy"][idx_a].astype(np.float64)
        pts_b = features[j]["xy"][idx_b].astype(np.float64)
        E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=_E_METHOD,
                                       prob=0.999, threshold=ransac_thresh)
        if E is None or mask is None:
            continue
        m = mask.ravel().astype(bool)
        if m.sum() < min_inliers:
            continue
        verified.append((i, j, idx_a[m], idx_b[m]))
        gaps.append(j - i)
    if verbose:
        if verified:
            g = np.array(gaps)
            print(f"  loop closures: {len(verified)}/{len(candidates)} candidate "
                  f"pairs verified (>= {min_inliers} inliers); frame-gap "
                  f"min={g.min()} median={int(np.median(g))} max={g.max()}, "
                  f"{sum(len(v[2]) for v in verified)} correspondences")
        else:
            print(f"  loop closures: 0/{len(candidates)} candidate pairs verified "
                  f"-- no revisit found (trajectory may not loop)")
    return verified


def match_loops(features, K, window=8, min_gap=None, top_k=6, n_words=256,
                wrap_span=None, min_inliers=30, verbose=True):
    """Full loop-closure front-end: BoW retrieval of distant revisits UNION
    sequential wrap candidates (head-block x tail-block) -> geometric
    verification. Returns verified (i,j,idx_i,idx_j) tuples to append to the
    windowed matches before build_tracks.

    min_gap defaults to window+1 so it only proposes pairs the windowed matcher
    could not already have linked. wrap_span (frames at each end considered for
    the sequential closure) defaults to max(2*window, n//6)."""
    n = len(features)
    if min_gap is None:
        min_gap = window + 1
    if n <= min_gap:
        if verbose:
            print(f"  loop closure skipped: only {n} frames (< min_gap {min_gap})")
        return []
    if wrap_span is None:
        wrap_span = max(2 * window, n // 6)
    t0 = time.time()

    vocab = build_vocabulary(features, n_words=n_words, verbose=verbose)
    if vocab is not None:
        bow = bow_vectors(features, vocab)
    else:
        if verbose:
            print("  BoW vocabulary unavailable -> mean-descriptor retrieval fallback")
        bow = _mean_descriptor_vectors(features)
    bow_cands = retrieve_candidates(bow, min_gap=min_gap, top_k=top_k)

    wrap = wrap_candidates(n, span=wrap_span, min_gap=min_gap)
    seen = set(bow_cands)
    candidates = list(bow_cands) + [p for p in wrap if p not in seen]
    if verbose:
        n_extra = len(candidates) - len(bow_cands)
        print(f"  candidates: {len(bow_cands)} appearance (BoW top_k={top_k}) + "
              f"{n_extra} sequential wrap (span={wrap_span}) = {len(candidates)} "
              f"(min_gap={min_gap})")

    verified = verify_pairs(features, K, candidates, min_inliers=min_inliers,
                            verbose=verbose)
    if verbose:
        print(f"  loop-closure matching in {time.time()-t0:.1f}s")
    return verified
