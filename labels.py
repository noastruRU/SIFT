"""
label_cameras.py — SIFT interactive camera labelling
=====================================================
Run this ONCE when you first set up the rig (or after moving cables).

It opens each camera one at a time, shows a live preview window so you
can physically see which camera activated, then asks you to type a label
for it (e.g. "top_left", "bottom_right", or just "top_0" .. "top_4").

The result is saved to camera_map.json which both configure_cameras.py
and jetson_backend.py load automatically.

Usage:
    python label_cameras.py            # interactive labelling
    python label_cameras.py --scan     # just print what cameras exist + port paths
    python label_cameras.py --show 3   # show live feed from physical index 3

Requirements:
    pip install pygrabber    (for stable device enumeration by port path)
    OpenCV (already installed)
"""

import argparse
import json
import sys
import time
import platform
from pathlib import Path

import cv2

CAMERA_MAP_FILE = Path(__file__).parent / "camera_map.json"
PREVIEW_DURATION = 999   # seconds — window stays open until user presses a key
N_TOP_DEFAULT = 5        # how many cameras are "top" (rest are "bottom")

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _ok(m):   print(f"  \033[92m✓\033[0m  {m}")
def _err(m):  print(f"  \033[91m✗\033[0m  {m}")
def _info(m): print(f"  \033[94m·\033[0m  {m}")
def _hdr(m):  print(f"\n\033[1m{m}\033[0m")
def _ask(m):  return input(f"\n  \033[93m?\033[0m  {m} ").strip()

# ---------------------------------------------------------------------------
# Device enumeration — returns (physical_index, name, port_path)
# ---------------------------------------------------------------------------

def enumerate_devices():
    """
    Returns list of dicts: {index, name, port_path}
    Uses pygrabber for names (stable), scans only as many indices as pygrabber found.
    """
    # ── pygrabber for names ───────────────────────────────────────────────
    names = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        _info(f"pygrabber found {len(names)} device name(s): {names}")
    except ImportError:
        _err("pygrabber not installed — pip install pygrabber")
        sys.exit(1)

    if not names:
        return []

    # ── Scan only the indices pygrabber knows about ───────────────────────
    # pygrabber index == cv2 DSHOW index (both use DirectShow enumeration)
    devices = []
    for idx, name in enumerate(names):
        _info(f"Probing index {idx} ('{name}')...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            devices.append({"index": idx, "name": name, "port_path": f"dshow:{idx}"})
            _ok(f"  index {idx} opens OK")
        else:
            cap.release()
            _err(f"  index {idx} could not open")
        time.sleep(0.2)

    return devices

# ---------------------------------------------------------------------------
# Open a camera (try both backends)
# ---------------------------------------------------------------------------

def open_cam(idx):
    """Try DSHOW first (matches pygrabber indices), fall back to MSMF."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None

# ---------------------------------------------------------------------------
# Show live preview until keypress
# ---------------------------------------------------------------------------

def show_preview(idx, title="Camera Preview"):
    cap = open_cam(idx)
    if cap is None:
        _err(f"Cannot open camera {idx}")
        return False

    window = f"{title} — cam{idx}  [press any key to continue]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    _info(f"Showing live preview for camera {idx} — press any key in the preview window")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Overlay the index prominently so you know which cam this is
        cv2.putText(frame, f"Physical index: {idx}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, title, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF != 255:   # any key pressed
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# ---------------------------------------------------------------------------
# cmd: scan
# ---------------------------------------------------------------------------

def cmd_scan():
    _hdr("Enumerating cameras...")
    devices = enumerate_devices()

    if not devices:
        _err("No cameras found. Check USB connections and close other apps.")
        return

    print()
    print(f"  {'IDX':>4}  {'NAME':<35}  PORT / DEVICE ID")
    print("  " + "─" * 75)
    for d in devices:
        print(f"  {d['index']:>4}  {d['name']:<35}  {d['port_path']}")
    print()
    print(f"  {len(devices)} camera(s) found")
    print()

    # Check if a map already exists
    if CAMERA_MAP_FILE.exists():
        _ok(f"Existing map found: {CAMERA_MAP_FILE}")
        existing = json.loads(CAMERA_MAP_FILE.read_text())
        print()
        for e in existing:
            print(f"    {e['label']:<12} [{e['group']:6}]  phys={e['physical']}  '{e['name']}'")
    else:
        _info("No camera_map.json yet — run without --scan to create one")

# ---------------------------------------------------------------------------
# cmd: label (interactive)
# ---------------------------------------------------------------------------

def cmd_label(n_top):
    _hdr("Interactive camera labelling")
    print()
    print("  This will open each camera one at a time.")
    print("  Look at which camera's LED lights up (or which feed appears),")
    print("  then type a label for it.")
    print()
    print("  Suggested labels:  top_0  top_1  top_2  top_3  top_4")
    print("                     bot_0  bot_1  bot_2  bot_3")
    print()

    devices = enumerate_devices()
    if not devices:
        _err("No cameras found.")
        sys.exit(1)

    print(f"  Found {len(devices)} camera(s): indices {[d['index'] for d in devices]}")
    print()

    # Load existing map if present (so we can resume / update)
    existing = {}
    if CAMERA_MAP_FILE.exists():
        try:
            for e in json.loads(CAMERA_MAP_FILE.read_text()):
                existing[e["physical"]] = e
            _info(f"Loaded existing map from {CAMERA_MAP_FILE} ({len(existing)} entries)")
        except Exception:
            pass

    entries = []

    for d in devices:
        idx  = d["index"]
        name = d["name"]
        port = d["port_path"]

        print(f"\n  ── Camera {idx}  '{name}' ──")
        _info(f"Port: {port}")

        if idx in existing:
            prev_label = existing[idx]["label"]
            _info(f"Previously labelled as: '{prev_label}'")
            keep = _ask(f"Keep label '{prev_label}'? [Y/n]")
            if keep.lower() in ("", "y", "yes"):
                entries.append(existing[idx])
                _ok(f"Kept: {prev_label}")
                continue

        # Show preview
        ans = _ask("Open live preview? [Y/n]")
        if ans.lower() in ("", "y", "yes"):
            show_preview(idx, f"Camera {idx} — '{name}'")

        # Ask for label
        while True:
            label = _ask(f"Label for camera {idx} (or 'skip'): ")
            if label.lower() == "skip":
                _info("Skipping")
                break
            if label:
                break
            print("    Label cannot be empty")

        if label.lower() == "skip":
            continue

        # Ask for group
        group_ans = _ask("Group — [T]op or [B]ottom? [T]").lower()
        group = "bottom" if group_ans in ("b", "bottom") else "top"

        # Is global reference?
        global_ans = _ask("Global reference camera? [y/N]").lower()
        is_global  = global_ans in ("y", "yes")

        entries.append({
            "physical":  idx,
            "label":     label,
            "group":     group,
            "is_global": is_global,
            "name":      name,
            "port_path": port,
        })
        _ok(f"Saved: {label} [{group}] → physical {idx}")

    if not entries:
        _err("No cameras labelled — nothing saved")
        return

    # Sort: top first (preserving label order), then bottom
    top    = [e for e in entries if e["group"] == "top"]
    bottom = [e for e in entries if e["group"] == "bottom"]

    # Re-assign logical labels cam_0, cam_1, ... in order
    ordered = top + bottom
    for i, e in enumerate(ordered):
        e["label"] = f"cam_{i}"

    # Ensure exactly one global reference
    globals_ = [e for e in ordered if e.get("is_global")]
    if not globals_ and ordered:
        ordered[0]["is_global"] = True
        _info(f"No global reference set — defaulting to {ordered[0]['label']}")

    CAMERA_MAP_FILE.write_text(json.dumps(ordered, indent=2))
    _ok(f"Saved {len(ordered)} entries to {CAMERA_MAP_FILE}")

    print()
    print("  Final mapping:")
    print(f"  {'LABEL':<10}  {'GROUP':<8}  {'PHYS':>5}  {'GLOBAL':>7}  NAME")
    print("  " + "─" * 65)
    for e in ordered:
        star = "  ★" if e.get("is_global") else ""
        print(f"  {e['label']:<10}  {e['group']:<8}  {e['physical']:>5}  {star:<7}  {e['name']}")
    print()
    print("  jetson_backend.py and configure_cameras.py will load this file automatically.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIFT interactive camera labeller")
    parser.add_argument("--scan",  action="store_true", help="List cameras and exit")
    parser.add_argument("--show",  type=int, metavar="IDX", help="Show live preview for one index")
    parser.add_argument("--n-top", type=int, default=N_TOP_DEFAULT,
                        help=f"Number of top cameras (default {N_TOP_DEFAULT})")
    args = parser.parse_args()

    print("=" * 55)
    print("  SIFT Camera Labeller")
    print("=" * 55)
    print(f"  Platform   : {platform.system()} {platform.release()}")
    print(f"  OpenCV     : {cv2.__version__}")
    print(f"  Map file   : {CAMERA_MAP_FILE}")
    print("=" * 55)

    if args.show is not None:
        show_preview(args.show)
    elif args.scan:
        cmd_scan()
    else:
        cmd_label(args.n_top)


if __name__ == "__main__":
    main()