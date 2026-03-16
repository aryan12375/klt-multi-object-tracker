"""
tracker.py — Core Multi-Object KLT Tracker
==========================================
Implements:
  - KLT tracking with Lucas-Kanade optical flow
  - Forward-Backward error checking (drops drifting points)
  - Kalman Filter per-object bounding box prediction
  - Adaptive feature selection (dynamic maxCorners)
  - YOLO initialization (optional hybrid mode)
  - DBSCAN clustering for multi-object association
  - Occlusion-aware track lifecycle management
  - Trail persistence with alpha blending
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

from models.kalman import KalmanBoxTracker
from utils.clustering import cluster_points_to_objects
from utils.features import adaptive_max_corners


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class Track:
    """Represents a single tracked object."""
    track_id: int
    points: np.ndarray          # current (N,1,2) float32 feature points
    bbox: tuple                 # (x, y, w, h)
    kalman: KalmanBoxTracker
    age: int = 0                # frames alive
    lost_frames: int = 0        # consecutive frames with no KLT match
    id_confirmed: bool = False  # True after MIN_HITS frames
    color: tuple = field(default_factory=lambda: None)
    trail: list = field(default_factory=list)  # list of centroid (x,y) for trail

    def centroid(self) -> tuple:
        x, y, w, h = self.bbox
        return (int(x + w / 2), int(y + h / 2))


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

class TrackerConfig:
    # Lucas-Kanade optical flow params
    LK_WIN_SIZE       = (15, 15)
    LK_MAX_LEVEL      = 2
    LK_CRITERIA       = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Shi-Tomasi feature detection
    ST_QUALITY        = 0.01
    ST_MIN_DIST       = 7
    ST_BLOCK_SIZE     = 7

    # Forward-Backward error threshold (pixels)
    FB_THRESHOLD      = 1.0

    # Clustering: DBSCAN
    DBSCAN_EPS        = 60.0    # max pixel distance for same cluster
    DBSCAN_MIN_PTS    = 8

    # Track lifecycle
    MAX_LOST_FRAMES   = 15      # Kalman predicts up to this many frames
    MIN_HITS          = 8       # frames before track is displayed
    MIN_POINTS        = 8     # minimum feature points to keep a track alive

    # Feature re-detection interval (frames)
    REDETECT_INTERVAL = 5

    # Trail settings
    TRAIL_MAX_LEN     = 20
    TRAIL_ALPHA       = 0.6

    # Colour palette (BGR) for track IDs
    PALETTE = [
        (0, 200, 255), (0, 255, 128), (255, 80, 80),
        (255, 200, 0), (180, 0, 255), (0, 180, 255),
        (255, 0, 160), (100, 255, 100), (255, 140, 0),
        (0, 255, 220), (200, 100, 255), (255, 255, 80),
    ]


# ──────────────────────────────────────────────
#  Main Tracker Class
# ──────────────────────────────────────────────

class KLTMultiTracker:
    def __init__(self, cfg: TrackerConfig = None, use_yolo: bool = False,
                 yolo_weights: str = None, yolo_cfg_path: str = None,
                 yolo_names: str = None):
        self.cfg       = cfg or TrackerConfig()
        self.tracks    = {}            # track_id → Track
        self._next_id  = 0
        self._frame_idx = 0
        self._prev_gray = None
        self._trail_canvas = None      # persistent canvas for trails

        # Metrics
        self.id_switch_count = 0
        self._prev_centroids = {}      # track_id → centroid at last frame

        # YOLO (optional)
        self.use_yolo = use_yolo
        self.yolo_net = None
        self.yolo_output_layers = []
        self.yolo_classes = []
        if use_yolo and yolo_weights and yolo_cfg_path:
            self._init_yolo(yolo_weights, yolo_cfg_path, yolo_names)

        self._lk_params = dict(
            winSize=self.cfg.LK_WIN_SIZE,
            maxLevel=self.cfg.LK_MAX_LEVEL,
            criteria=self.cfg.LK_CRITERIA,
            flags=0,
            minEigThreshold=1e-3,
        )

    # ── YOLO initializer ──────────────────────

    def _init_yolo(self, weights, cfg_path, names_path):
        try:
            self.yolo_net = cv2.dnn.readNet(weights, cfg_path)
            layer_names = self.yolo_net.getLayerNames()
            out_idx = self.yolo_net.getUnconnectedOutLayers()
            self.yolo_output_layers = [layer_names[i - 1] for i in out_idx.flatten()]
            if names_path:
                with open(names_path) as f:
                    self.yolo_classes = [l.strip() for l in f.readlines()]
            print("[YOLO] Loaded successfully.")
        except Exception as e:
            print(f"[YOLO] Could not load model: {e}. Falling back to Shi-Tomasi init.")
            self.yolo_net = None

    def _yolo_detect(self, frame):
        """Returns list of (x,y,w,h) bounding boxes from YOLO."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(self.yolo_output_layers)
        boxes, confidences = [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = np.argmax(scores)
                conf   = float(scores[cls_id])
                if conf > 0.4:
                    cx, cy, bw, bh = (det[:4] * np.array([w, h, w, h])).astype(int)
                    boxes.append([cx - bw//2, cy - bh//2, bw, bh])
                    confidences.append(conf)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        if len(indices) == 0:
            return []
        return [boxes[i] for i in indices.flatten()]

    # ── Feature detection ─────────────────────

    def _detect_features(self, gray: np.ndarray, mask: np.ndarray = None,
                          bbox: tuple = None) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners, optionally inside a bbox."""
        roi = gray
        offset = (0, 0)
        if bbox is not None:
            x, y, w, h = [max(0, v) for v in bbox]
            x2 = min(gray.shape[1], x + w)
            y2 = min(gray.shape[0], y + h)
            if x2 <= x or y2 <= y:
                return None
            roi = gray[y:y2, x:x2]
            offset = (x, y)

        max_corners = adaptive_max_corners(roi)
        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=max_corners,
            qualityLevel=self.cfg.ST_QUALITY,
            minDistance=self.cfg.ST_MIN_DIST,
            blockSize=self.cfg.ST_BLOCK_SIZE,
        )
        if pts is None:
            return None
        # Shift back to full-frame coords
        pts[:, 0, 0] += offset[0]
        pts[:, 0, 1] += offset[1]
        return pts

    # ── Forward-Backward error check ──────────

    def _fb_filter(self, prev_gray, curr_gray, pts_fwd, pts_back) -> np.ndarray:
        """Return boolean mask: True = reliable point."""
        fb_err = np.linalg.norm(pts_fwd.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
        return fb_err < self.cfg.FB_THRESHOLD

    # ── KLT flow with FB check ─────────────────

    def _track_points(self, prev_gray, curr_gray,
                      points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (good_new, good_old) after forward-backward filtering.
        """
        if points is None or len(points) == 0:
            return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.float32)

        # Forward pass
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None, **self._lk_params)

        if p1 is None:
            return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.float32)

        # Backward pass
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, p1, None, **self._lk_params)

        # Status mask
        fwd_ok  = (st_fwd.ravel() == 1)
        back_ok = (st_back.ravel() == 1)

        # FB error mask
        fb_ok = self._fb_filter(points, p0_back, points, p0_back)

        good_mask = fwd_ok & back_ok & fb_ok
        if good_mask.sum() == 0:
            return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.float32)

        return p1[good_mask], points[good_mask]

    # ── Bbox from points ──────────────────────

    @staticmethod
    def _bbox_from_points(pts: np.ndarray) -> tuple:
        xy = pts.reshape(-1, 2)
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        pad = 12
        return (int(x1) - pad, int(y1) - pad,
                int(x2 - x1) + 2*pad, int(y2 - y1) + 2*pad)

    # ── Initialise new tracks ─────────────────

    def _init_track(self, points: np.ndarray, bbox: tuple) -> Track:
        tid   = self._next_id
        color = self.cfg.PALETTE[tid % len(self.cfg.PALETTE)]
        kf    = KalmanBoxTracker(bbox)
        track = Track(track_id=tid, points=points, bbox=bbox,
                      kalman=kf, color=color)
        self._next_id += 1
        return track

    def _bootstrap_tracks(self, gray: np.ndarray, frame: np.ndarray):
        """First-frame initialisation using YOLO or global feature clustering."""
        if self.use_yolo and self.yolo_net:
            bboxes = self._yolo_detect(frame)
            for bbox in bboxes:
                pts = self._detect_features(gray, bbox=bbox)
                if pts is not None and len(pts) >= self.cfg.MIN_POINTS:
                    t = self._init_track(pts, bbox)
                    self.tracks[t.track_id] = t
            if self.tracks:
                return

        # Fallback: global Shi-Tomasi + clustering
        pts = self._detect_features(gray)
        if pts is None or len(pts) < self.cfg.MIN_POINTS:
            return
        clusters = cluster_points_to_objects(
            pts.reshape(-1, 2), self.cfg.DBSCAN_EPS, self.cfg.DBSCAN_MIN_PTS)
        for cluster_pts in clusters:
            arr = cluster_pts.reshape(-1, 1, 2).astype(np.float32)
            bbox = self._bbox_from_points(arr)
            t = self._init_track(arr, bbox)
            self.tracks[t.track_id] = t

    # ── Main update ───────────────────────────

    def update(self, frame: np.ndarray) -> dict:
        """
        Process one frame. Returns dict of active Track objects.
        Call this every frame in your video loop.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self._trail_canvas is None:
            self._trail_canvas = np.zeros_like(frame, dtype=np.uint8)

        # ── Bootstrap on first frame ──────────
        if self._prev_gray is None:
            self._bootstrap_tracks(gray, frame)
            self._prev_gray = gray
            self._frame_idx += 1
            return self.tracks

        # ── Track existing objects ────────────
        ids_to_delete = []
        all_new_points_flat = []  # for duplicate suppression

        for tid, track in self.tracks.items():
            if track.points is not None and len(track.points) >= 1:
                new_pts, old_pts = self._track_points(
                    self._prev_gray, gray, track.points)
            else:
                new_pts = np.empty((0, 1, 2), dtype=np.float32)

            if len(new_pts) >= self.cfg.MIN_POINTS:
                track.points     = new_pts
                track.bbox       = self._bbox_from_points(new_pts)
                track.kalman.update(track.bbox)
                track.lost_frames = 0
                track.age        += 1
                if track.age >= self.cfg.MIN_HITS:
                    track.id_confirmed = True

                # Update trail
                cx, cy = track.centroid()
                track.trail.append((cx, cy))
                if len(track.trail) > self.cfg.TRAIL_MAX_LEN:
                    track.trail.pop(0)

                all_new_points_flat.extend(new_pts.reshape(-1, 2).tolist())
            else:
                # Lost KLT — let Kalman predict
                track.lost_frames += 1
                predicted_bbox = track.kalman.predict()
                if predicted_bbox is not None:
                    track.bbox = tuple(predicted_bbox.astype(int))
                track.points = None  # no tracked points this frame

                if track.lost_frames > self.cfg.MAX_LOST_FRAMES:
                    ids_to_delete.append(tid)
                    continue

                # Try re-detecting features inside predicted bbox
                re_pts = self._detect_features(gray, bbox=track.bbox)
                if re_pts is not None and len(re_pts) >= self.cfg.MIN_POINTS:
                    track.points = re_pts
                    # Count ID switch if centroid jumped too far
                    if tid in self._prev_centroids:
                        dist = np.linalg.norm(
                            np.array(track.centroid()) - np.array(self._prev_centroids[tid]))
                        if dist > 80:
                            self.id_switch_count += 1

        for tid in ids_to_delete:
            del self.tracks[tid]

        # ── Periodic global re-detection for new objects ──
        if self._frame_idx % self.cfg.REDETECT_INTERVAL == 0:
            self._detect_new_objects(gray, all_new_points_flat)

        # ── Update previous centroids for ID switch tracking ──
        self._prev_centroids = {
            tid: t.centroid() for tid, t in self.tracks.items()
        }

        self._prev_gray = gray
        self._frame_idx += 1
        return self.tracks

    def _detect_new_objects(self, gray, existing_pts_flat):
        """Globally re-detect and spawn new tracks for un-tracked objects."""
        pts_all = self._detect_features(gray)
        if pts_all is None:
            return

        if existing_pts_flat:
            existing = np.array(existing_pts_flat)
            new_flat = pts_all.reshape(-1, 2)
            # Keep only points far from any existing tracked point
            novel = []
            for p in new_flat:
                dists = np.linalg.norm(existing - p, axis=1)
                if dists.min() > self.cfg.DBSCAN_EPS:
                    novel.append(p)
            if not novel:
                return
            pts_novel = np.array(novel).reshape(-1, 1, 2).astype(np.float32)
        else:
            pts_novel = pts_all

        clusters = cluster_points_to_objects(
            pts_novel.reshape(-1, 2), self.cfg.DBSCAN_EPS, self.cfg.DBSCAN_MIN_PTS)
        for cluster_pts in clusters:
            arr = cluster_pts.reshape(-1, 1, 2).astype(np.float32)
            bbox = self._bbox_from_points(arr)
            t = self._init_track(arr, bbox)
            self.tracks[t.track_id] = t
