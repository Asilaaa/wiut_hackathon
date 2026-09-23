"""Scene layout of the fixed camera: crossings, stop lines, traffic lights, islands.

Geometry comes from scene.json (hand-annotated) in 1920x1080 reference pixels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SCENE_JSON = Path(__file__).with_name("scene.json")


def _poly(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


def inside(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def side_of_line(line: np.ndarray, x: float, y: float) -> float:
    """Signed side of point (x, y) w.r.t. the directed line p0 -> p1 (>0 left, <0 right in image coords)."""
    (x0, y0), (x1, y1) = line
    return float((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0))


@dataclass
class Scene:
    crosswalks: dict[str, np.ndarray]
    stop_lines: dict[str, np.ndarray]
    stop_line_signal: dict[str, str]
    lights: dict[str, dict[str, tuple[int, int, int, int]]]
    islands: dict[str, np.ndarray]
    carriageway_heading: dict[str, float]

    @classmethod
    def load(cls, path: Path = SCENE_JSON) -> "Scene":
        d = json.loads(path.read_text())
        return cls(
            crosswalks={k: _poly(v) for k, v in d["crosswalks"].items()},
            stop_lines={k: _poly(v["line"]) for k, v in d["stop_lines"].items()},
            stop_line_signal={k: v["signal"] for k, v in d["stop_lines"].items()},
            lights={k: {lamp: tuple(box) for lamp, box in v.items() if not lamp.startswith("_")}
                    for k, v in d["traffic_lights"].items()},
            islands={k: _poly(v) for k, v in d["non_carriageway"].items() if not k.startswith("_")},
            carriageway_heading={k: float(v["heading_deg"]) for k, v in d["carriageways"].items()},
        )

    def to_video(self, A: np.ndarray) -> "Scene":
        """This scene mapped into a video's own pixel coordinates, given A: video -> reference (see align.py)."""
        Ainv = cv2.invertAffineTransform(A)

        def pts(p: np.ndarray) -> np.ndarray:
            return (p @ Ainv[:, :2].T + Ainv[:, 2]).astype(np.float32)

        def box(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            (cx, cy), = pts(np.float32([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]))
            hw, hh = (b[2] - b[0]) / 2, (b[3] - b[1]) / 2
            return int(round(cx - hw)), int(round(cy - hh)), int(round(cx + hw)), int(round(cy + hh))

        return Scene(
            crosswalks={k: pts(v) for k, v in self.crosswalks.items()},
            stop_lines={k: pts(v) for k, v in self.stop_lines.items()},
            stop_line_signal=dict(self.stop_line_signal),
            lights={k: {lamp: box(b) for lamp, b in v.items()} for k, v in self.lights.items()},
            islands={k: pts(v) for k, v in self.islands.items()},
            carriageway_heading=dict(self.carriageway_heading),
        )

    def crosswalk_at(self, x: float, y: float) -> str | None:
        for name, poly in self.crosswalks.items():
            if inside(poly, x, y):
                return name
        return None

    def on_island(self, x: float, y: float) -> bool:
        return any(inside(p, x, y) for p in self.islands.values())

    def draw(self, img: np.ndarray) -> np.ndarray:
        """Overlay the layout on a reference-resolution frame (for checking and for the website)."""
        out = img.copy()
        layer = img.copy()
        for poly in self.crosswalks.values():
            cv2.fillPoly(layer, [poly.astype(np.int32)], (255, 200, 0))
        for poly in self.islands.values():
            cv2.fillPoly(layer, [poly.astype(np.int32)], (180, 0, 255))
        out = cv2.addWeighted(layer, 0.45, out, 0.55, 0)
        for name, poly in self.crosswalks.items():
            cv2.polylines(out, [poly.astype(np.int32)], True, (255, 200, 0), 2)
            x, y = poly.mean(0).astype(int)
            cv2.putText(out, f"crosswalk {name}", (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for name, line in self.stop_lines.items():
            p0, p1 = line.astype(int)
            cv2.line(out, tuple(p0), tuple(p1), (0, 0, 255), 3)
            cv2.putText(out, f"stop line {name}", (int(p0[0]), int(p0[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        for name, lamps in self.lights.items():
            for lamp, (x0, y0, x1, y1) in lamps.items():
                color = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0)}[lamp]
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
            x0, y0 = next(iter(lamps.values()))[:2]
            cv2.putText(out, f"light {name}", (x0 - 20, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return out
