"""Draw cached trajectories on a still frame of the video, coloured by direction of travel.

    python scripts/plot_tracks.py samples/C3905.MP4
Writes outputs/tracks/<video>.jpg. The colour wheel in the corner maps hue to heading.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.tracking import load_tracks  # noqa: E402
from src.video import iter_frames  # noqa: E402

MIN_TRACK_PX = 60  # ignore tracks that barely move (parked cars, standing people)


def heading_color(dx: float, dy: float) -> tuple[int, int, int]:
    hue = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)  # OpenCV hue is 0..179
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_wheel(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    for deg in range(0, 360, 5):
        a = math.radians(deg)
        cv2.line(img, (cx, cy), (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))),
                 heading_color(math.cos(a), math.sin(a)), 3)
    cv2.putText(img, "heading", (cx - r, cy + r + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    args = ap.parse_args()

    name = Path(args.video).name
    dets = load_tracks(config.CACHE_DIR / f"{name}.tracks.csv")
    _, _, bg = next(iter_frames(args.video))
    canvas = (bg * 0.45).astype(np.uint8)

    tracks: dict[int, list] = defaultdict(list)
    types: dict[int, str] = {}
    for d in dets:
        # bottom-centre of the box = where the road user touches the ground
        tracks[d.track_id].append(((d.x1 + d.x2) / 2, d.y2))
        types[d.track_id] = d.type

    for tid, pts in tracks.items():
        pts = np.array(pts)
        if len(pts) < 5 or np.linalg.norm(pts[-1] - pts[0]) < MIN_TRACK_PX:
            continue
        is_person = types[tid] == "person"
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            color = (200, 200, 200) if is_person else heading_color(x1 - x0, y1 - y0)
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color, 1 if is_person else 2, cv2.LINE_AA)

    draw_wheel(canvas, config.REF_W - 90, 90, 60)
    cv2.putText(canvas, f"{name}: vehicles coloured by heading, pedestrians grey", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out = config.ROOT / "outputs" / "tracks" / f"{Path(name).stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
