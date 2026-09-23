"""Run detection + tracking over videos once and cache the tracks to cache/<video>.tracks.csv.

    python scripts/cache_tracks.py samples/*.MP4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.tracking import Tracker, save_tracks  # noqa: E402
from src.video import iter_frames, probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--stride", type=int, default=config.DETECT_STRIDE)
    ap.add_argument("--max-sec", type=float, default=None, help="only process the first N seconds (for quick tests)")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = ap.parse_args()

    for video in args.videos:
        out = config.CACHE_DIR / f"{Path(video).name}.tracks.csv"
        if out.exists() and not args.force:
            print(f"[{Path(video).name}] cached -> {out}")
            continue
        info = probe(video)
        tracker = Tracker()
        dets, t0 = [], time.perf_counter()
        for idx, t, frame in iter_frames(video, args.stride):
            if args.max_sec is not None and t > args.max_sec:
                break
            dets += tracker.update(frame, idx, t)
            if idx % (args.stride * 100) == 0:
                el = time.perf_counter() - t0
                print(f"[{Path(video).name}] {t:6.1f}/{info.duration:.0f}s  {el:6.0f}s elapsed  "
                      f"speed {t / max(el, 1e-6):.2f}x realtime", flush=True)
        save_tracks(dets, out)
        n_tracks = Counter(d.type for d in {(d.track_id, d.type): d for d in dets}.values())
        meta = {"video": Path(video).name, "fps": info.fps, "n_frames": info.n_frames, "duration": info.duration,
                "width": info.width, "height": info.height, "stride": args.stride,
                "detector": config.DETECTOR_WEIGHTS.name, "imgsz": config.DETECT_IMGSZ,
                "runtime_sec": round(time.perf_counter() - t0, 1), "tracks_by_type": dict(n_tracks)}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"[{Path(video).name}] {len(dets)} detections, tracks {dict(n_tracks)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
