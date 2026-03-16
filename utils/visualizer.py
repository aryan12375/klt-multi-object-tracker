"""
utils/visualizer.py — Rendering Engine
=======================================
Handles all drawing on frames:
  - Bounding boxes with track ID labels
  - Smooth alpha-blended motion trails
  - Feature point dots
  - Kalman-predicted bbox (dashed) when track is lost
  - HUD overlay (FPS, track count, metrics)
  - Status bar at bottom
"""

import cv2
import numpy as np
from typing import Dict


class Visualizer:
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.45
    FONT_MED   = 0.6
    FONT_LARGE = 0.75
    THICKNESS  = 1

    def __init__(self, trail_alpha: float = 0.6):
        self.trail_alpha = trail_alpha
        self._trail_canvas = None

    def _ensure_canvas(self, frame: np.ndarray):
        if self._trail_canvas is None or self._trail_canvas.shape != frame.shape:
            self._trail_canvas = np.zeros_like(frame, dtype=np.uint8)

    def draw_trail(self, frame: np.ndarray, trail: list, color: tuple) -> np.ndarray:
        """
        Draw a smooth fading motion trail.
        Older segments are more transparent.
        """
        if len(trail) < 2:
            return frame

        n = len(trail)
        for i in range(1, n):
            alpha  = (i / n) * self.trail_alpha
            pt1    = tuple(map(int, trail[i-1]))
            pt2    = tuple(map(int, trail[i]))
            thick  = max(1, int(3 * (i / n)))
            overlay = frame.copy()
            cv2.line(overlay, pt1, pt2, color, thick, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def draw_track(self, frame: np.ndarray, track, is_lost: bool = False) -> np.ndarray:
        """Draw bbox, label, and feature points for one track."""
        x, y, w, h = track.bbox
        color = track.color

        # Ensure bbox is in frame
        fh, fw = frame.shape[:2]
        x  = max(0, x);  y  = max(0, y)
        x2 = min(fw, x + w); y2 = min(fh, y + h)

        if is_lost:
            # Dashed bbox for Kalman-predicted (lost) tracks
            frame = self._draw_dashed_rect(frame, (x, y, x2-x, y2-y), color)
            label = f"ID:{track.track_id} [pred]"
        else:
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"ID:{track.track_id} (age:{track.age})"

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SMALL, 1)
        lx, ly = x, max(0, y - 4)
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 2),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (lx + 3, ly - 2),
                    self.FONT, self.FONT_SMALL, (0, 0, 0), 1, cv2.LINE_AA)

        # Feature points
        if track.points is not None:
            for pt in track.points.reshape(-1, 2):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3,
                           color, cv2.FILLED, cv2.LINE_AA)

        return frame

    def _draw_dashed_rect(self, frame, bbox, color, gap=8):
        """Draw a dashed rectangle."""
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        pts = [(x, y, x2, y), (x2, y, x2, y2),
               (x2, y2, x, y2), (x, y2, x, y)]
        for x1l, y1l, x2l, y2l in pts:
            # Draw dashed line segment
            dx = x2l - x1l; dy = y2l - y1l
            length = max(int(np.hypot(dx, dy)), 1)
            steps  = length // gap
            for s in range(steps):
                if s % 2 == 0:
                    t1 = s / steps; t2 = (s + 1) / steps
                    p1 = (int(x1l + t1*dx), int(y1l + t1*dy))
                    p2 = (int(x1l + t2*dx), int(y1l + t2*dy))
                    cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        return frame

    def draw_hud(self, frame: np.ndarray, overlay_lines: list) -> np.ndarray:
        """
        Draw the top-left HUD panel with metric lines.
        overlay_lines: list of (text, color) from MetricsTracker.
        """
        panel_h = 18 * len(overlay_lines) + 12
        panel_w = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        for i, (text, color) in enumerate(overlay_lines):
            y_pos = 16 + i * 18
            cv2.putText(frame, text, (8, y_pos),
                        self.FONT, self.FONT_SMALL, color, 1, cv2.LINE_AA)
        return frame

    def draw_status_bar(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Draw a thin status bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 22), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, text, (8, h - 6),
                    self.FONT, self.FONT_SMALL, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def render(self, frame: np.ndarray, tracks: dict,
               overlay_lines: list, status: str = "") -> np.ndarray:
        """
        Full render pass — call once per frame.
        Draws: trails → tracks → HUD → status bar.
        """
        out = frame.copy()

        # Trails (drawn first, so boxes render on top)
        for tid, track in tracks.items():
            if track.trail:
                out = self.draw_trail(out, track.trail, track.color)

        # Tracks
        for tid, track in tracks.items():
            if not track.id_confirmed:
                continue  # suppress unconfirmed tracks
            is_lost = (track.points is None or len(track.points) == 0)
            out = self.draw_track(out, track, is_lost=is_lost)

        # HUD
        out = self.draw_hud(out, overlay_lines)

        # Status bar
        if status:
            out = self.draw_status_bar(out, status)

        return out
