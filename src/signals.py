"""Traffic-light state from fixed lamp regions of the frame."""
from __future__ import annotations

import cv2
import numpy as np

LAMP_HUE = {"red": ((0, 12), (165, 180)), "yellow": ((12, 35),), "green": ((40, 95),)}


def lamp_score(frame: np.ndarray, box: tuple[int, int, int, int], lamp: str) -> float:
    """Fraction of strongly coloured pixels of the lamp's own hue inside its box.

    Saturation, not brightness, separates a lit lamp: in direct sun the grey housing is
    brighter than the LED, but it is colourless.
    """
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, bool)
    for lo, hi in LAMP_HUE[lamp]:
        hue_ok |= (h >= lo) & (h < hi)
    lit = hue_ok & (s > 120) & (v > 60)
    return float(lit.mean())


def light_state(frame: np.ndarray, lamps: dict[str, tuple[int, int, int, int]], min_score: float = 0.03) -> str:
    """'red' / 'yellow' / 'green' for the brightest lamp, or 'unknown' if none is lit."""
    scores = {lamp: lamp_score(frame, box, lamp) for lamp, box in lamps.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_score else "unknown"
