"""
models/kalman.py — Kalman Filter for Bounding Box Prediction
=============================================================
State vector: [cx, cy, w, h, vx, vy, vw, vh]
Measures:     [cx, cy, w, h]

A constant-velocity model. When KLT loses points, we call predict()
each frame to estimate where the bbox should be. When points come back,
we call update() to correct the filter.
"""

import numpy as np


class KalmanBoxTracker:
    """
    Per-object Kalman filter that tracks a bounding box as
    [cx, cy, w, h] with a constant-velocity motion model.
    """

    def __init__(self, bbox: tuple):
        """
        bbox: (x, y, w, h) — top-left corner + size
        """
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = _build_kalman_filter()
        cx, cy, w, h = self._to_cxcywh(bbox)
        self.kf.statePre = np.array(
            [[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self._last_prediction = None

    @staticmethod
    def _to_cxcywh(bbox):
        x, y, w, h = bbox
        return float(x + w/2), float(y + h/2), float(w), float(h)

    @staticmethod
    def _to_xywh(cx, cy, w, h):
        return (int(cx - w/2), int(cy - h/2), int(w), int(h))

    def predict(self) -> np.ndarray:
        """Advance the filter one step; returns (x, y, w, h) or None."""
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        w, h = max(w, 10), max(h, 10)
        result = np.array(self._to_xywh(cx, cy, w, h), dtype=np.float32)
        self._last_prediction = result
        return result

    def update(self, bbox: tuple):
        """Correct the filter with a new measurement."""
        cx, cy, w, h = self._to_cxcywh(bbox)
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)


def _build_kalman_filter():
    """Construct a 8-state, 4-measurement Kalman filter."""
    kf = _CV2KF(dynamParams=8, measureParams=4)

    # Transition matrix  (constant velocity)
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement matrix  (we only observe cx,cy,w,h)
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1
    kf.measurementMatrix[3, 3] = 1

    # Process noise
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.processNoiseCov[4:, 4:] *= 10   # higher uncertainty in velocity

    # Measurement noise
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

    # Posterior error
    kf.errorCovPost = np.eye(8, dtype=np.float32)

    return kf


class _CV2KF:
    """Thin wrapper around cv2.KalmanFilter to avoid import at top-level."""
    def __init__(self, dynamParams, measureParams):
        import cv2
        self._kf = cv2.KalmanFilter(dynamParams, measureParams)

    def __getattr__(self, name):
        return getattr(self._kf, name)

    def __setattr__(self, name, value):
        if name == '_kf':
            super().__setattr__(name, value)
        else:
            try:
                setattr(self._kf, name, value)
            except AttributeError:
                super().__setattr__(name, value)

    def predict(self):
        return self._kf.predict()

    def correct(self, measurement):
        return self._kf.correct(measurement)
