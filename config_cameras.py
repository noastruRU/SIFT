"""
configure_cameras.py — SIFT one-time camera setup
===================================================
No external dependencies beyond OpenCV.

Usage:
    python configure_cameras.py --scan          # show all cameras: index, name, brightness
    python configure_cameras.py                 # apply settings to all found cameras
    python configure_cameras.py --index 0 2 5   # apply settings to specific indices only
    python configure_cameras.py --warmup 10     # more warmup frames if cameras are slow

Workflow:
    1. Run --scan  →  note which index each physical camera shows up as
    2. Fill in CAMERA_MAP in jetson_backend.py using those indices
    3. Run without --scan to apply the right settings to every camera
"""

import argparse
import platform
import sys
import time

import cv2

# ---------------------------------------------------------------------------
# Per-camera-type settings
# Profile is chosen by case-insensitive substring match on the device name
# returned by Windows (e.g. "innomaker" matches "Innomaker-U20CAM-1080p-S1").
# "default" is used when nothing matches.
#
# CAP_PROP_AUTO_EXPOSURE on MSMF/DSHOW:  3 = auto,  1 = manual
# ---------------------------------------------------------------------------

CAMERA_PROFILES = {
    "Innomaker-U20CAM-1080p-S1": {
        cv2.CAP_PROP_AUTO_EXPOSURE: 1,
        cv2.CAP_PROP_AUTO_WB:       1,
        cv2.CAP_PROP_BRIGHTNESS:    15,
        cv2.CAP_PROP_CONTRAST:      64,
        cv2.CAP_PROP_SATURATION:    100,
        cv2.CAP_PROP_SHARPNESS:     6,
        cv2.CAP_PROP_GAMMA:         100,
        cv2.CAP_PROP_BACKLIGHT:     31,
    },
    "usb camera": {
        cv2.CAP_PROP_AUTO_EXPOSURE: 1,
        cv2.CAP_PROP_AUTO_WB:       1,
        cv2.CAP_PROP_BRIGHTNESS:    45,
        cv2.CAP_PROP_CONTRAST:      43,
        cv2.CAP_PROP_SATURATION:    64,
        cv2.CAP_PROP_SHARPNESS:     14,
        cv2.CAP_PROP_GAMMA:         100,
        cv2.CAP_PROP_BACKLIGHT:     80,
    },
    "default": {
        cv2.CAP_PROP_AUTO_EXPOSURE: 3,
        cv2.CAP_PROP_AUTO_WB:       1,
        cv2.CAP_PROP_BRIGHTNESS:    15,
        cv2.CAP_PROP_CONTRAST:      43,
        cv2.CAP_PROP_SATURATION:    100,
        cv2.CAP_PROP_SHARPNESS:     6,
        cv2.CAP_PROP_GAMMA:         100,
        cv2.CAP_PROP_BACKLIGHT:     31,
    },
}

PROP_NAMES = {
    cv2.CAP_PROP_AUTO_EXPOSURE: "AUTO_EXPOSURE",
    cv2.CAP_PROP_AUTO_WB:       "AUTO_WB",
    cv2.CAP_PROP_BRIGHTNESS:    "BRIGHTNESS",
    cv2.CAP_PROP_CONTRAST:      "CONTRAST",
    cv2.CAP_PROP_SATURATION:    "SATURATION",
    cv2.CAP_PROP_SHARPNESS:     "SHARPNESS",
    cv2.CAP_PROP_GAMMA:         "GAMMA",
    cv2.CAP_PROP_BACKLIGHT:     "BACKLIGHT",
}

WARMUP_FRAMES = 5    # frames to grab before applying settings (wakes MSMF driver)
OPEN_RETRIES  = 2
RETRY_DELAY   = 0.8
SCAN_MAX_IDX  = 15   # scan indices 0 through this value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(m):   print(f"  \033[92m✓\033[0m  {m}")
def _err(m):  print(f"  \033[91m✗\033[0m  {m}")
def _info(m): print(f"  \033[94m·\033[0m  {m}")
def _hdr(m):  print(f"\n\033[1m{m}\033[0m")


def _profile_for(name: str) -> tuple:
    """Return (key, props) by substring match on camera name."""
    nl = name.lower()
    for key, props in CAMERA_PROFILES.items():
        if key == "default":
            continue
        if key.lower() in nl:
            return key, props
    return "default", CAMERA_PROFILES["default"]


def _open(idx: int):
    """Try DSHOW then MSMF. Returns (cap, backend_name) or (None, None)."""
    for backend, bname in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            return cap, bname
        cap.release()
    return None, None


def _get_device_name(idx: int) -> str:
    """
    Try to read the camera's friendly name via pygrabber.
    Falls back to a generic string if not installed.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        return names[idx] if idx < len(names) else f"Camera {idx}"
    except Exception:
        return f"Camera {idx}"

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def cmd_scan():
    _hdr(f"Scanning camera indices 0–{SCAN_MAX_IDX}...")
    print()
    print(f"  {'IDX':>4}  {'NAME':<35}  {'PROFILE':<10}  {'BRIGHTNESS':>10}  BACKEND")
    print("  " + "─" * 75)

    found = []
    consecutive_missing = 0

    for idx in range(SCAN_MAX_IDX + 1):
        cap, backend = _open(idx)
        if cap is None:
            consecutive_missing += 1
            if idx > 8 and consecutive_missing >= 3:
                break   # stop well past the last camera
            continue

        consecutive_missing = 0
        name = _get_device_name(idx)
        profile_key, _ = _profile_for(name)

        # grab a few frames for a real brightness reading
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok:
                frame = f
        mean = f"{frame.mean():.1f}" if frame is not None else "no frame"
        cap.release()
        time.sleep(0.2)

        print(f"  {idx:>4}  {name:<35}  {profile_key:<10}  {mean:>10}  {backend}")
        found.append(idx)

    print()
    if found:
        print(f"  Found {len(found)} camera(s) at indices: {found}")
        print()
        print("  → Fill in CAMERA_MAP in jetson_backend.py using these indices.")
    else:
        _err("No cameras found. Check USB connections and close other apps.")

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

def cmd_configure(indices: list, warmup: int):
    _hdr(f"Configuring cameras at indices: {indices}")

    total_ok = total_fail = total_skipped = 0

    for idx in indices:
        _hdr(f"Camera {idx}")

        # Open with retries
        cap = bname_used = None
        for attempt in range(OPEN_RETRIES):
            cap, bname_used = _open(idx)
            if cap:
                break
            _err(f"Open attempt {attempt+1}/{OPEN_RETRIES} failed — retrying…")
            time.sleep(RETRY_DELAY)

        if not cap:
            _err(f"Could not open index {idx} — skipping")
            total_skipped += 1
            continue

        name = _get_device_name(idx)
        profile_key, props = _profile_for(name)
        _info(f"Opened via {bname_used}  |  name: '{name}'  |  profile: '{profile_key}'")

        # Warmup frames — wakes the MSMF/DSHOW driver so CAP_PROP_* writes stick
        _info(f"Grabbing {warmup} warmup frame(s)…")
        grabbed = 0
        for _ in range(warmup):
            ok_r, _ = cap.read()
            if ok_r:
                grabbed += 1
        if grabbed == 0:
            _err("No frames readable — camera may be in use by another app")
            cap.release()
            total_skipped += 1
            continue
        _info(f"Driver ready ({grabbed}/{warmup} frames grabbed)")

        # Write and verify each property
        cam_ok = cam_fail = 0
        for prop, target in props.items():
            pname = PROP_NAMES.get(prop, str(prop))
            cap.set(prop, target)
            actual = cap.get(prop)
            if abs(actual - target) < 1.5:
                _ok(f"{pname} = {target}  (reads {actual:.0f})")
                cam_ok += 1
            else:
                _err(f"{pname} = {target}  → got {actual:.1f}  (not supported by this camera)")
                cam_fail += 1

        total_ok   += cam_ok
        total_fail += cam_fail
        print(f"  cam{idx}: {cam_ok} props applied, {cam_fail} not supported")

        cap.release()
        time.sleep(0.4)

    print()
    configured = len(indices) - total_skipped
    print(f"Done — {total_ok} props set across {configured} camera(s) "
          f"({total_skipped} skipped), {total_fail} not supported by hardware")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIFT one-time camera configurator")
    parser.add_argument("--scan",    action="store_true",
                        help="List all cameras with index, name, brightness")
    parser.add_argument("--index",   nargs="+", type=int, metavar="N",
                        help="Configure only these physical indices (default: all found)")
    parser.add_argument("--warmup",  type=int, default=WARMUP_FRAMES,
                        help=f"Warmup frames before writing settings (default: {WARMUP_FRAMES})")
    args = parser.parse_args()

    print("=" * 52)
    print("  SIFT Camera Configurator")
    print("=" * 52)
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  OpenCV   : {cv2.__version__}")
    print("=" * 52)

    if args.scan:
        cmd_scan()
        return

    # Configure — discover indices unless --index was given explicitly
    if args.index:
        indices = args.index
    else:
        # Auto-discover by scanning first
        _hdr("Discovering cameras…")
        indices = []
        consecutive_missing = 0
        for idx in range(SCAN_MAX_IDX + 1):
            cap, _ = _open(idx)
            if cap:
                cap.release()
                indices.append(idx)
                consecutive_missing = 0
                _info(f"Found camera at index {idx}")
            else:
                consecutive_missing += 1
                if idx > 8 and consecutive_missing >= 3:
                    break
            time.sleep(0.1)

        if not indices:
            _err("No cameras found. Check USB connections.")
            sys.exit(1)

    cmd_configure(indices, args.warmup)
    print("\n\033[92mDone. You can now start jetson_backend.py.\033[0m\n")


if __name__ == "__main__":
    main()