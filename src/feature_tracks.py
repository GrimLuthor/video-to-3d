"""Windowed multi-view feature matching + track building.

This is the piece the original chain_poses.py front-end was missing, and the
actual fix for the scale drift on longer_walk.mp4. chain_poses only ever
matched frame i -> frame i+1, so:
  * every 3D point was triangulated from the *narrowest possible* baseline
    (adjacent frames), giving each link a weakly-constrained scale, and
  * the bundle-adjustment graph was a pure chain -- no observation ever tied
    a frame to a non-adjacent one, so relative scale between distant cameras
    was an almost-flat direction of the cost and BA literally could not see
    the drift.

Here we instead match each frame against the next W frames (a "sequential
window"), geometrically verify every pair (essential-matrix RANSAC), and join
the surviving matches transitively into multi-frame tracks via union-find. A
point on a facade you walk past is now a single track spanning many frames
with a *wide* baseline between its first and last observation -- exactly the
constraint that pins relative scale and kills drift, no loop closure needed.

Outputs feed incremental_sfm.py.
"""

import cv2
import numpy as np


def extract_features(frames, contrast_threshold=0.04, nfeatures=6000):
    """SIFT keypoints + descriptors per frame.

    Returns a list of dicts: {"xy": (N,2) float32 pixel coords, "desc": (N,128)
    float32 descriptors, "kp": raw cv2 keypoints}. `nfeatures` caps each frame
    to its strongest keypoints -- full-HD frames otherwise yield 10-15k, which
    makes the O(n^2) all-pairs windowed matching below far too slow for no real
    gain (the strongest few thousand give plenty of well-spread tracks).
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold)
    features = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        kp, desc = sift.detectAndCompute(gray, None)
        desc = _root_sift(desc)
        xy = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
        features.append({"xy": xy, "desc": desc, "kp": kp})
    return features


def _root_sift(desc, eps=1e-7):
    """RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise each descriptor
    then take the element-wise sqrt, so Euclidean distance approximates the
    Hellinger kernel. A near-free, consistent matching improvement over raw
    SIFT."""
    if desc is None or len(desc) == 0:
        return desc
    desc = desc / (desc.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(desc).astype(np.float32)


_FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
# MAGSAC++ (USAC) gives more accurate essential matrices with fewer outliers than
# plain RANSAC -> cleaner relative poses -> fewer repeated-structure ghosts.
_E_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def _ratio_dict(desc_q, desc_t, ratio_thresh):
    """query-index -> train-index for matches passing the Lowe ratio test."""
    out = {}
    raw = _FLANN.knnMatch(np.asarray(desc_q, np.float32), np.asarray(desc_t, np.float32), k=2)
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            out[m.queryIdx] = m.trainIdx
    return out


def _match_pair(desc_a, desc_b, ratio_thresh=0.75):
    """Lowe-ratio FLANN match with a MUTUAL (cross-check) constraint: a match
    (a_i, b_j) is kept only if a_i's best in b is b_j AND b_j's best in a is a_i.
    Cross-check removes many one-sided false matches cheaply before the
    essential-matrix geometric verification."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    ab = _ratio_dict(desc_a, desc_b, ratio_thresh)   # a_i -> b_j
    ba = _ratio_dict(desc_b, desc_a, ratio_thresh)   # b_j -> a_i
    idx_a, idx_b = [], []
    for ai, bj in ab.items():
        if ba.get(bj, -1) == ai:                     # mutual
            idx_a.append(ai)
            idx_b.append(bj)
    return np.array(idx_a, int), np.array(idx_b, int)


def match_windowed(features, K, window=8, ratio_thresh=0.75, ransac_thresh=1.0,
                   min_inliers=20, verbose=True):
    """Match every frame i to frames i+1 .. i+window, geometrically verify each
    pair with an essential-matrix RANSAC (using K), and return the surviving
    (frame_i, frame_j, idx_i, idx_j) inlier correspondences.

    `window` is the key knob vs. the old chain (which was effectively window=1):
    it must be large enough that a feature you walk past is captured in several
    overlapping pairs (long tracks, wide baselines) but small enough that
    frames i and i+window still share appearance.
    """
    n = len(features)
    verified = []  # (i, j, idx_i inliers, idx_j inliers)
    n_attempted = 0
    for i in range(n):
        for j in range(i + 1, min(i + window + 1, n)):
            n_attempted += 1
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
    if verbose:
        n_inl = sum(len(v[2]) for v in verified)
        print(f"  windowed match: {len(verified)}/{n_attempted} pairs verified "
              f"(window={window}), {n_inl} inlier correspondences total")
    return verified


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, int)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_tracks(features, verified, min_track_len=2, verbose=True):
    """Join verified pairwise matches transitively into multi-frame tracks.

    Each (frame, keypoint) is a node; every verified correspondence unions two
    nodes. Connected components are tracks. Tracks that observe the *same*
    frame twice are internally inconsistent (a feature can't be two points in
    one image) and are discarded wholesale -- the standard COLMAP-style guard
    against transitive mismatches corrupting a track.

    Returns a list of tracks, each a dict {frame_idx: (x, y)}, sorted by frame.
    """
    n_per = [len(f["xy"]) for f in features]
    offset = np.concatenate([[0], np.cumsum(n_per)])
    total = int(offset[-1])
    uf = _UnionFind(total)

    for (i, j, idx_i, idx_j) in verified:
        base_i, base_j = offset[i], offset[j]
        for a, b in zip(idx_i, idx_j):
            uf.union(base_i + a, base_j + b)

    # group node ids by root
    roots = np.array([uf.find(x) for x in range(total)])
    order = np.argsort(roots)
    tracks = []
    n_conflicts = 0
    start = 0
    roots_sorted = roots[order]
    for end in range(1, total + 1):
        if end == total or roots_sorted[end] != roots_sorted[start]:
            members = order[start:end]
            start = end
            if len(members) < min_track_len:
                continue
            obs = {}
            conflict = False
            for node in members:
                frame = int(np.searchsorted(offset, node, side="right") - 1)
                kp_idx = int(node - offset[frame])
                if frame in obs:
                    conflict = True
                    break
                obs[frame] = tuple(features[frame]["xy"][kp_idx])
            if conflict:
                n_conflicts += 1
                continue
            if len(obs) >= min_track_len:
                tracks.append(dict(sorted(obs.items())))

    if verbose:
        lens = np.array([len(t) for t in tracks])
        spans = np.array([max(t) - min(t) for t in tracks]) if tracks else np.zeros(0)
        print(f"  tracks: {len(tracks)} (discarded {n_conflicts} frame-conflicting), "
              f"len mean={lens.mean():.2f} max={lens.max() if len(lens) else 0}, "
              f"3+ views={(lens >= 3).sum()} ({(lens >= 3).mean():.1%}), "
              f"frame-span mean={spans.mean():.1f} max={spans.max() if len(spans) else 0}")
    return tracks
