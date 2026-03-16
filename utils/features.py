"""
utils/features.py — Adaptive Feature Selection
================================================
Computes a dynamic maxCorners value for cv2.goodFeaturesToTrack
based on the region area and scene texture complexity (Laplacian variance).

Logic:
  - Larger bbox  → more corners needed to cover the object
  - More texture → allow more corners (high-detail region)
  - Small/empty  → reduce to avoid noise
"""

import cv2
import numpy as np


# Tunable bounds
MIN_CORNERS = 8
MAX_CORNERS = 150
BASE_DENSITY = 1 / 800.0   # 1 corner per 800 px² baseline


def adaptive_max_corners(roi: np.ndarray) -> int:
    """
    Given a grayscale region-of-interest, return a dynamic maxCorners value.

    Steps:
      1. Compute area-based baseline count.
      2. Scale by normalised Laplacian variance (texture measure).
      3. Clamp to [MIN_CORNERS, MAX_CORNERS].
    """
    if roi is None or roi.size == 0:
        return MIN_CORNERS

    h, w = roi.shape[:2]
    area = max(h * w, 1)

    # Area-based baseline
    baseline = int(area * BASE_DENSITY)

    # Texture complexity: Laplacian variance
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    variance = lap.var()
    # Normalise: low variance → scale ~0.5, high variance → scale up to 2.0
    texture_scale = float(np.clip(variance / 200.0, 0.5, 2.0))

    corners = int(baseline * texture_scale)
    return int(np.clip(corners, MIN_CORNERS, MAX_CORNERS))
