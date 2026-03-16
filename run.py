"""
run.py — Main Entry Point
==========================
Usage:
    # Basic (webcam)
    python run.py

    # Video file
    python run.py --input path/to/video.mp4

    # Video file + save output
    python run.py --input video.mp4 --output tracked_output.mp4

    # YOLO hybrid initialisation
    python run.py --input video.mp4 --yolo \
        --yolo-weights yolov4-tiny.weights \
        --yolo-cfg    yolov4-tiny.cfg \
        --yolo-names  coco.names

    # With MOT17 ground truth (evaluation mode)
    python run.py --input video.mp4 --gt-file gt.txt

Controls:
    Q / ESC  → quit
    P        → pause/resume
    R        → reset all tracks
    S        → save current frame as PNG
"""

import cv2
import sys
import argparse
import time
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tracker import KLTMultiTracker, TrackerConfig
from utils.metrics import MetricsTracker
from utils.visualizer import Visualizer


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Enhanced KLT Multi-Object Tracker")
    p.add_argument("--input",        default=0,    help="Video path or 0 for webcam")
    p.add_argument("--output",       default=None, help="Output video path (optional)")
    p.add_argument("--width",  type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--yolo",         action="store_true")
    p.add_argument("--yolo-weights", default="yolov4-tiny.weights")
    p.add_argument("--yolo-cfg",     default="yolov4-tiny.cfg")
    p.add_argument("--yolo-names",   default="coco.names")
    p.add_argument("--gt-file",      default=None, help="MOT17 GT annotation file")
    p.add_argument("--no-display",   action="store_true", help="Headless mode")
    return p.parse_args()


# ──────────────────────────────────────────────
#  MOT17 GT loader
# ──────────────────────────────────────────────

def load_mot17_gt(path: str) -> dict:
    """
    Parse MOT17 ground truth file.
    Format: frame, id, x, y, w, h, conf, class, visibility
    Returns: {frame_id: [(x,y,w,h), ...]}
    """
    gt = {}
    if path is None or not os.path.exists(path):
        return gt
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            bbox = (float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5]))
            gt.setdefault(frame_id, []).append(bbox)
    print(f"[GT] Loaded {sum(len(v) for v in gt.values())} annotations "
          f"across {len(gt)} frames.")
    return gt


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    input_src = int(args.input) if str(args.input).isdigit() else args.input

    # ── Video capture ─────────────────────────
    cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_src}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source: {input_src}  {actual_w}×{actual_h} @ {src_fps:.1f} fps")

    # ── Output writer ─────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps,
                                 (actual_w, actual_h))
        print(f"[INFO] Writing to {args.output}")

    # ── Components ────────────────────────────
    cfg     = TrackerConfig()
    tracker = KLTMultiTracker(
        cfg=cfg,
        use_yolo=args.yolo,
        yolo_weights=args.yolo_weights,
        yolo_cfg_path=args.yolo_cfg,
        yolo_names=args.yolo_names,
    )
    metrics  = MetricsTracker()
    viz      = Visualizer(trail_alpha=cfg.TRAIL_ALPHA)
    gt_data  = load_mot17_gt(args.gt_file)

    paused       = False
    frame_idx    = 0
    screenshot_n = 0

    print("[INFO] Running. Press Q/ESC to quit, P to pause, R to reset, S to screenshot.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video.")
                break

            metrics.tick()
            frame_idx += 1

            # ── Tracker update ────────────────
            tracks = tracker.update(frame)

            # ── GT evaluation (if available) ──
            if gt_data and frame_idx in gt_data:
                pred_boxes = [t.bbox for t in tracks.values() if t.id_confirmed]
                metrics.update_gt(pred_boxes, gt_data[frame_idx])
                metrics.set_id_switches(tracker.id_switch_count)

            # ── Confirmed track count ─────────
            confirmed = sum(1 for t in tracks.values() if t.id_confirmed)

            # ── Build overlay text ────────────
            overlay_lines = metrics.get_overlay_lines(
                confirmed, tracker.id_switch_count)

            # ── Status bar ────────────────────
            mode = "YOLO+KLT" if args.yolo else "Shi-Tomasi+KLT"
            status = (f"Mode:{mode}  |  FB-Check:ON  |  Kalman:ON  |  "
                      f"AdaptiveFeatures:ON  |  Frame:{frame_idx}")

            # ── Render ────────────────────────
            out_frame = viz.render(frame, tracks, overlay_lines, status)

        # ── Display ───────────────────────────
        if not args.no_display:
            cv2.imshow("KLT Multi-Object Tracker [CSE-3144]", out_frame)

        if writer:
            writer.write(out_frame)

        # ── Key handling ──────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):           # Q or ESC
            break
        elif key == ord("p"):               # Pause/resume
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}]")
        elif key == ord("r"):               # Reset tracks
            tracker = KLTMultiTracker(cfg=cfg)
            metrics.reset()
            print("[RESET] Tracks cleared.")
        elif key == ord("s"):               # Screenshot
            fname = f"screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(fname, out_frame)
            print(f"[SAVED] {fname}")
            screenshot_n += 1

    # ── Cleanup ───────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Final metrics report ──────────────────
    report = metrics.summary()
    report["id_switches"] = tracker.id_switch_count
    print("\n" + "="*50)
    print("  FINAL METRICS REPORT")
    print("="*50)
    for k, v in report.items():
        print(f"  {k:<20} {v}")
    print("="*50)


if __name__ == "__main__":
    main()
