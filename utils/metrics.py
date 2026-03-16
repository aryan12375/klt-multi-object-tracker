"""
utils/metrics.py — MOT Metrics Calculator
==========================================
Implements:
  - IoU (Intersection over Union) between two bboxes
  - MOTP (Multiple Object Tracking Precision) — mean IoU across matched pairs
  - MOTA (Multiple Object Tracking Accuracy):
        MOTA = 1 − (FP + FN + IDSW) / GT
  - Per-frame FPS tracking
  - ID Switch counter (maintained by tracker, reported here)
  - Active track count

Usage (evaluation mode):
  Pass ground-truth MOT17 annotations to MetricsTracker.update_gt()
  after each frame and call .summary() at the end.

Usage (demo mode with no GT):
  Just call .update_fps() each frame and .get_overlay_text() for display.
"""

import time
import numpy as np
from collections import defaultdict


def iou(boxA: tuple, boxB: tuple) -> float:
    """
    Compute IoU between two (x, y, w, h) bounding boxes.
    Returns float in [0, 1].
    """
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class MetricsTracker:
    """
    Tracks running MOT metrics. Supports both GT-available and GT-free modes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._frame_times = []
        self._last_time   = time.perf_counter()

        # MOT metrics (require GT)
        self.total_gt        = 0
        self.total_fp        = 0
        self.total_fn        = 0
        self.id_switches     = 0
        self.iou_sum         = 0.0
        self.matched_pairs   = 0
        self._frame_count    = 0

        # Per-frame tracking
        self.fps_history     = []  # rolling window

    # ── FPS ──────────────────────────────────

    def tick(self):
        """Call once per frame to record timing."""
        now = time.perf_counter()
        dt  = now - self._last_time
        self._last_time = now
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        self._frame_count += 1

    def current_fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return float(np.mean(self.fps_history[-10:]))

    # ── GT-based metrics ─────────────────────

    def update_gt(self, pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5):
        """
        Match predictions to GT boxes using IoU greedy matching.
        pred_boxes / gt_boxes: list of (x, y, w, h)
        """
        matched_gt   = set()
        matched_pred = set()

        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pb in enumerate(pred_boxes):
            for j, gb in enumerate(gt_boxes):
                iou_matrix[i, j] = iou(pb, gb)

        # Greedy match by descending IoU
        for _ in range(min(len(pred_boxes), len(gt_boxes))):
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] < iou_threshold:
                break
            matched_pred.add(i)
            matched_gt.add(j)
            self.iou_sum      += iou_matrix[i, j]
            self.matched_pairs += 1
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gt_boxes)   - len(matched_gt)
        self.total_fp += max(fp, 0)
        self.total_fn += max(fn, 0)
        self.total_gt += len(gt_boxes)

    def set_id_switches(self, count: int):
        self.id_switches = count

    # ── Computed properties ───────────────────

    @property
    def mota(self) -> float:
        if self.total_gt == 0:
            return 0.0
        return 1.0 - (self.total_fp + self.total_fn + self.id_switches) / self.total_gt

    @property
    def motp(self) -> float:
        if self.matched_pairs == 0:
            return 0.0
        return self.iou_sum / self.matched_pairs

    # ── Overlay text for display ──────────────

    def get_overlay_lines(self, active_tracks: int, id_switches: int) -> list:
        """
        Returns list of (text, color) tuples for cv2.putText overlay.
        """
        fps  = self.current_fps()
        lines = [
            (f"FPS: {fps:.1f}",              (0, 255, 128)),
            (f"Tracks: {active_tracks}",     (0, 200, 255)),
            (f"ID Switches: {id_switches}",  (255, 200, 80)),
            (f"Frame: {self._frame_count}",  (180, 180, 180)),
        ]
        if self.matched_pairs > 0:
            lines += [
                (f"MOTA: {self.mota:.3f}", (100, 255, 100)),
                (f"MOTP: {self.motp:.3f}", (100, 200, 255)),
            ]
        return lines

    def summary(self) -> dict:
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        return {
            "frames"      : self._frame_count,
            "avg_fps"     : round(avg_fps, 2),
            "mota"        : round(self.mota, 4),
            "motp"        : round(self.motp, 4),
            "total_fp"    : self.total_fp,
            "total_fn"    : self.total_fn,
            "id_switches" : self.id_switches,
        }
