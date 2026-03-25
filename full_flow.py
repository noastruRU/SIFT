"""
capture_and_inspect.py — Capture from USB cameras, then run SIFT inspection

Usage:
    python capture_and_inspect.py
    python capture_and_inspect.py --cameras 0 1 2 3
    python capture_and_inspect.py --cameras 0 1 2 3 --warmup-frames 30
    python capture_and_inspect.py --mode optimal --out my_results
    python capture_and_inspect.py --save-captures          # keep the captured images

Output:
    • Prints per-nut results to stdout
    • Saves  inference_results/overview_cam{N}.png  (annotated views)
    • Saves  inference_results/nut_grid.png          (crop grid)
    • Saves  inference_results/summary.json
    • Optionally saves  inference_results/capture_cam{N}.jpg
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image
import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL = "https://noa-strupinsky-sift.hf.space"
INSPECT_URL  = f"{HF_SPACE_URL}/inspect"
TIMEOUT      = 300

CANVAS_SIZE  = 256

# Camera warm-up: number of frames to discard before the real capture.
# Auto-exposure on most USB webcams converges within 10–30 frames.
# Raise this if images are still overexposed; lower it to speed things up.
DEFAULT_WARMUP_FRAMES = 20

# Resolution to request from each camera (set to None to use device default)
CAPTURE_WIDTH  = 1280
CAPTURE_HEIGHT = 960


# ══════════════════════════════════════════════════════════════════════════════
# Camera capture
# ══════════════════════════════════════════════════════════════════════════════

def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera and configure resolution."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. "
                           "Check it is connected and not in use by another process.")
    if CAPTURE_WIDTH and CAPTURE_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    return cap



def capture_frame(cap: cv2.VideoCapture, cam_index: int) -> np.ndarray:
    """Grab a single frame after warm-up."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Camera {cam_index} failed to deliver a frame.")
    return frame


def capture_all_cameras(
    camera_indices: list,
    warmup_frames: int,
    out_dir: Path,
    save_captures: bool,
) -> list:
    """
    Capture one frame from each camera **sequentially** — open, warm up, capture,
    release — before moving to the next camera.

    This avoids USB bandwidth exhaustion when all cameras share the same hub,
    since only one camera is active at any given moment.
    """
    print(f"\nCapturing from {len(camera_indices)} camera(s) sequentially: {camera_indices}")
    out_dir.mkdir(parents=True, exist_ok=True)

    captured_paths = []
    for idx in camera_indices:
        print(f"\n  Camera {idx}: opening …")
        cap = open_camera(idx)

        # Warm up: drain frames so auto-exposure / AWB can settle
        print(f"  Camera {idx}: warming up ({warmup_frames} frames) …", end="", flush=True)
        for i in range(warmup_frames):
            ok, _ = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Camera {idx} stopped responding during warm-up "
                    f"(frame {i}/{warmup_frames}). "
                    "Try reducing --warmup-frames or checking the USB connection.")
            time.sleep(0.03)
        print(" done")

        # Capture the real frame
        frame = capture_frame(cap, idx)
        cap.release()
        print(f"  Camera {idx}: released")

        if save_captures:
            path = out_dir / f"capture_cam{idx}.jpg"
        else:
            path = out_dir / f"_tmp_cam{idx}.jpg"

        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        label = str(path) if save_captures else f"{path} (temp)"
        print(f"  Camera {idx}: saved → {label}")
        captured_paths.append(str(path))

    print(f"\nAll {len(camera_indices)} cameras captured.\n")
    return captured_paths


def cleanup_temp_files(paths: list, save_captures: bool):
    """Remove temp capture files if the user didn't ask to keep them."""
    if not save_captures:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Call HF Space
# ══════════════════════════════════════════════════════════════════════════════

def call_hf_space(image_paths: list, mode: str, sensitivity: float) -> dict:
    files  = []
    opened = []
    for p in image_paths:
        f = open(p, "rb")
        opened.append(f)
        files.append(("images", (Path(p).name, f, "image/jpeg")))

    data = {
        "mode":               mode,
        "target_sensitivity": str(sensitivity),
    }

    print(f"Sending {len(image_paths)} image(s) to {INSPECT_URL} …")
    try:
        resp = requests.post(INSPECT_URL, files=files, data=data, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. The pipeline may still be loading models.")
        print("       Try again — subsequent requests are faster.")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {HF_SPACE_URL}")
        sys.exit(1)
    except requests.exceptions.HTTPError:
        print(f"ERROR: Server returned {resp.status_code}")
        print(resp.text[:1000])
        sys.exit(1)
    finally:
        for f in opened:
            f.close()

    return parse_html_response(resp.text)


# ══════════════════════════════════════════════════════════════════════════════
# Parse HTML response
# ══════════════════════════════════════════════════════════════════════════════

def parse_html_response(html: str) -> dict:
    overview_b64 = re.findall(
        r'<img src="data:image/png;base64,([^"]+)"[^>]*max-height:260px', html)
    crop_b64 = re.findall(
        r'<img src="data:image/png;base64,([^"]+)"[^>]*width:110px', html)
    nut_ids = [int(x) for x in re.findall(r'<strong>Nut (\d+)</strong>', html)]
    fused_scores = [float(x) for x in re.findall(
        r'Fused score:.*?<strong[^>]*>([\d.]+)</strong>', html)]
    decisions    = re.findall(r'(🔴 DEFECT|🟢 GOOD)', html)
    threshold_match = re.search(r'threshold: ([\d.]+)', html)
    threshold    = float(threshold_match.group(1)) if threshold_match else None
    summary_match = re.search(
        r'(⚠ \d+ / \d+ nuts DEFECTIVE|✓ All \d+ nuts GOOD)', html)
    summary_text = summary_match.group(1) if summary_match else "Unknown"

    nuts = []
    for i, nut_id in enumerate(nut_ids):
        nuts.append({
            "nut_id":     nut_id,
            "fused":      fused_scores[i] if i < len(fused_scores) else None,
            "is_anomaly": decisions[i] == "🔴 DEFECT" if i < len(decisions) else None,
            "decision":   "DEFECT" if (i < len(decisions) and decisions[i] == "🔴 DEFECT") else "GOOD",
        })

    return {
        "summary":      summary_text,
        "threshold":    threshold,
        "nuts":         nuts,
        "overview_b64": overview_b64,
        "crop_b64":     crop_b64,
        "n_nuts":       len(nuts),
        "n_defects":    sum(1 for n in nuts if n["is_anomaly"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def b64_to_img(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    arr  = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_outputs(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, b64 in enumerate(data["overview_b64"]):
        img  = b64_to_img(b64)
        path = out_dir / f"overview_cam{i}.png"
        cv2.imwrite(str(path), img)
        print(f"  Saved {path}")

    crops  = [b64_to_img(b) for b in data["crop_b64"]]
    n_nuts = data["n_nuts"]
    n_cams = len(data["overview_b64"])
    cell_sz = CANVAS_SIZE + 10

    if crops and n_nuts > 0:
        grid = np.ones((cell_sz * n_nuts, cell_sz * max(n_cams, 1), 3),
                       dtype=np.uint8) * 30
        for row, nut in enumerate(data["nuts"]):
            color = (0, 0, 220) if nut["is_anomaly"] else (0, 180, 0)
            for col in range(n_cams):
                idx = row * n_cams + col
                if idx >= len(crops):
                    break
                crop = crops[idx]
                y0, x0 = row * cell_sz, col * cell_sz
                h, w   = crop.shape[:2]
                grid[y0:y0+h, x0:x0+w] = crop
                cv2.rectangle(grid, (x0, y0),
                              (x0 + CANVAS_SIZE - 1, y0 + CANVAS_SIZE - 1), color, 3)
        grid_path = out_dir / "nut_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"  Saved {grid_path}")

    summary = {
        "threshold": data["threshold"],
        "n_nuts":    data["n_nuts"],
        "n_defects": data["n_defects"],
        "nuts": [
            {"nut_id": n["nut_id"], "fused_score": n["fused"], "decision": n["decision"]}
            for n in data["nuts"]
        ],
    }
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════

def print_results(data: dict):
    print()
    print("=" * 55)
    print("INSPECTION RESULTS")
    print("=" * 55)
    print(f"{'Nut':<8} {'Fused Score':<15} Decision")
    print("-" * 55)
    for n in data["nuts"]:
        decision = "⚠  DEFECT" if n["is_anomaly"] else "✓  GOOD"
        score    = f"{n['fused']:.4f}" if n["fused"] is not None else "N/A"
        print(f"nut_{n['nut_id']:<4} {score:<15} {decision}")
    print()
    print(f"SUMMARY:   {data['summary']}")
    print(f"Threshold: {data['threshold']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture from USB cameras and run SIFT inspection via HF Space")

    parser.add_argument(
        "--cameras", nargs="+", type=int, default=[0, 1, 2, 3],
        help="Camera device indices (default: 0 1 2 3)")
    parser.add_argument(
        "--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES,
        help=f"Frames to discard for auto-exposure warm-up (default: {DEFAULT_WARMUP_FRAMES}). "
             "Increase if images are still overexposed.")
    parser.add_argument(
        "--mode", default="target_recall",
        choices=["optimal", "target_recall", "max_recall"],
        help="Threshold mode (default: target_recall)")
    parser.add_argument(
        "--sensitivity", type=float, default=0.95,
        help="Target sensitivity for target_recall mode (default: 0.95)")
    parser.add_argument(
        "--out", default="inference_results",
        help="Output directory (default: inference_results)")
    parser.add_argument(
        "--save-captures", action="store_true",
        help="Keep captured camera images in the output directory")
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="Probe which camera indices are available and exit")

    args = parser.parse_args()

    # ── List available cameras and exit ──────────────────────────────────────
    if args.list_cameras:
        print("Probing camera indices 0–9 …")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  [{i}] available  ({int(w)}×{int(h)})")
                cap.release()
            else:
                print(f"  [{i}] not found")
        sys.exit(0)

    print(f"Cameras:       {args.cameras}")
    print(f"Warmup frames: {args.warmup_frames}")
    print(f"Mode:          {args.mode}")
    print(f"Sensitivity:   {args.sensitivity}")
    print(f"Space URL:     {HF_SPACE_URL}")

    out_dir = Path(args.out)

    # ── Step 1: Capture ───────────────────────────────────────────────────────
    captured_paths = capture_all_cameras(
        camera_indices=args.cameras,
        warmup_frames=args.warmup_frames,
        out_dir=out_dir,
        save_captures=args.save_captures,
    )

    # ── Step 2: Send to HF Space ──────────────────────────────────────────────
    data = call_hf_space(captured_paths, args.mode, args.sensitivity)

    # ── Step 3: Clean up temp files ───────────────────────────────────────────
    cleanup_temp_files(captured_paths, args.save_captures)

    # ── Step 4: Report & save ─────────────────────────────────────────────────
    print_results(data)
    save_outputs(data, out_dir)
    print(f"Done. Results saved to: {args.out}/")