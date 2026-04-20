"""
SIFT - Jetson Local Backend
============================
Simple sequential camera capture:
  Phase 1: open cameras in TOP group one at a time, wait for brightness < TARGET_MEAN, capture
  Phase 2: open cameras in BOT group one at a time, wait for brightness < TARGET_MEAN, capture

Camera settings are NOT applied here — run configure_cameras.py once after plugging in cameras.

Endpoints:
  GET    /health            - liveness + capture state
  GET    /config-status     - static reminder to run configure_cameras.py
  GET    /status            - full state (done cams, current cam, logs)
  GET    /scan              - probe indices 0-15, report which open + brightness
  GET    /image/{idx}       - return captured PNG for camera idx
  GET    /camera-map        - return current camera mapping
  POST   /camera-map/rebuild - re-resolve NAME_ORDER after plug/unplug
  POST   /capture/top       - start capturing top cameras in background
  POST   /capture/bottom    - start capturing bottom cameras in background
  POST   /inspect           - forward all images to HF Space, return results
  DELETE /session           - reset everything
  GET    /                  - serve sift.html
"""

import io
import os
import threading
import time
import webbrowser
import httpx
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import List

import cv2
import numpy as np

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ===========================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        CAMERA CONFIGURATION                              ║
# ║                                                                           ║
# ║  Run  python configure_cameras.py --scan  to see which physical index    ║
# ║  each camera shows up as, then fill in the table below.                  ║
# ║                                                                           ║
# ║  Each row:  physical_index  →  logical_label,  group,  is_global         ║
# ║    physical_index : integer shown by --scan / Windows device manager     ║
# ║    logical_label  : filename sent to HF Space (must match homographies)  ║
# ║    group          : "top" (captured first) or "bottom" (after blocker)   ║
# ║    is_global      : True for exactly one camera — the reference view     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

#            physical   label     group      global
CAMERA_MAP = [
    dict(physical=0, label="cam_2", group="top",    is_global=True ),   # ← edit me
    dict(physical=1, label="cam_1", group="bottom",    is_global=False),
    dict(physical=2, label="cam_0", group="bottom",    is_global=False),
    dict(physical=3, label="cam_4", group="top",    is_global=False),
    dict(physical=4, label="cam_6", group="top",    is_global=False),
    dict(physical=5, label="cam_8", group="bottom", is_global=False),
    dict(physical=6, label="cam_3", group="top", is_global=False),
    dict(physical=7, label="cam_7", group="bottom", is_global=False),
    dict(physical=8, label="cam_5", group="top", is_global=False),
]

# ===========================================================================

TARGET_MEAN = 200     # keep reading frames until mean brightness drops below this
MAX_FRAMES  = 100     # give up after this many frames and capture anyway
FRAME_SLEEP = 0.03    # seconds between frame reads (~33fps poll rate)

# ---------------------------------------------------------------------------
# Camera map helpers
# ---------------------------------------------------------------------------

_cam_map_lock = threading.Lock()
_cam_map: List[dict] = [dict(e) for e in CAMERA_MAP]   # mutable copy


def _top_cams() -> List[int]:
    with _cam_map_lock:
        return [e["physical"] for e in _cam_map if e["group"] == "top"]


def _bot_cams() -> List[int]:
    with _cam_map_lock:
        return [e["physical"] for e in _cam_map if e["group"] == "bottom"]


def _all_cams() -> List[int]:
    with _cam_map_lock:
        return [e["physical"] for e in _cam_map]


def _label_for(physical_idx: int) -> str:
    with _cam_map_lock:
        for e in _cam_map:
            if e["physical"] == physical_idx:
                return e["label"]
    return f"cam_{physical_idx}"


# ---------------------------------------------------------------------------
# Config status — static
# ---------------------------------------------------------------------------

_config_status = {
    "done":   True,
    "logs":   ["Camera settings applied by configure_cameras.py (run once after plug-in)."],
    "errors": [],
}

# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------

class Status(str, Enum):
    IDLE      = "idle"
    CAPTURING = "capturing"
    DONE      = "done"
    ERROR     = "error"

_state = {
    "status":      Status.IDLE,
    "phase":       None,
    "current_cam": None,
    "done_cams":   [],
    "failed_cams": [],
    "errors":      [],
    "logs":        [],
    "images":      {},
}
_lock = threading.Lock()


def _log(msg: str):
    print(msg, flush=True)
    with _lock:
        _state["logs"].append(msg)


def _set(**kwargs):
    with _lock:
        _state.update(kwargs)

# ---------------------------------------------------------------------------
# Core capture
# ---------------------------------------------------------------------------

def capture_one(cam_idx: int) -> bytes:
    _log(f"cam{cam_idx}: opening...")
    # Try DSHOW first (matches pygrabber indices), fall back to MSMF
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"cam{cam_idx}: cannot open")

    _log(f"cam{cam_idx}: open, waiting for brightness < {TARGET_MEAN}...")
    frame   = None
    settled = False

    try:
        for f in range(MAX_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"cam{cam_idx}: read failed at frame {f}")
            mean = frame.mean()
            if f % 20 == 0:
                _log(f"  cam{cam_idx} frame {f:>3}: mean={mean:.1f}")
            if mean < TARGET_MEAN:
                _log(f"cam{cam_idx}: settled after {f} frames (mean={mean:.1f})")
                settled = True
                break
            time.sleep(FRAME_SLEEP)

        if not settled:
            _log(f"cam{cam_idx}: never settled after {MAX_FRAMES} frames — capturing anyway")
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError(f"cam{cam_idx}: no frame captured")

    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"cam{cam_idx}: imencode failed")

    _log(f"cam{cam_idx}: captured (mean={frame.mean():.1f})")
    return buf.tobytes()

# ---------------------------------------------------------------------------
# Background capture thread
# ---------------------------------------------------------------------------

def _run_group(cam_list: list, phase: str):
    _set(phase=phase, status=Status.CAPTURING, current_cam=None)
    _log(f"\n=== Phase: {phase} — cameras {cam_list} ===")

    for cam_idx in cam_list:
        _set(current_cam=cam_idx)
        try:
            png = capture_one(cam_idx)
            with _lock:
                _state["images"][cam_idx] = png
                _state["done_cams"].append(cam_idx)
            _log(f"cam{cam_idx}: stored")
        except Exception as e:
            msg = f"cam{cam_idx}: ERROR — {e}"
            _log(msg)
            with _lock:
                _state["failed_cams"].append(cam_idx)
                _state["errors"].append(msg)

    _set(current_cam=None, status=Status.DONE)
    _log(f"=== {phase} done. captured={_state['done_cams']} failed={_state['failed_cams']} ===\n")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title="SIFT Jetson Backend", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class InspectRequest(BaseModel):
    hf_url: str
    mode: str                 = "target_recall"
    target_sensitivity: float = 0.95

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    with _lock:
        return {
            "status":      "ok",
            "capture":     _state["status"],
            "phase":       _state["phase"],
            "current_cam": _state["current_cam"],
            "done_cams":   list(_state["done_cams"]),
        }


@app.get("/config-status")
def config_status():
    return _config_status


@app.get("/camera-map")
def get_camera_map():
    with _cam_map_lock:
        mapping  = list(_cam_map)
        top_cams = [e["physical"] for e in _cam_map if e["group"] == "top"]
        bot_cams = [e["physical"] for e in _cam_map if e["group"] == "bottom"]
        all_cams = [e["physical"] for e in _cam_map]
    return {
        "mapping":  mapping,
        "top_cams": top_cams,
        "bot_cams": bot_cams,
        "all_cams": all_cams,
    }


@app.get("/status")
def status():
    with _lock:
        return {
            "status":      _state["status"],
            "phase":       _state["phase"],
            "current_cam": _state["current_cam"],
            "done_cams":   list(_state["done_cams"]),
            "failed_cams": list(_state["failed_cams"]),
            "errors":      list(_state["errors"]),
            "logs":        list(_state["logs"]),
            "cached_cams": list(_state["images"].keys()),
            "top_cams":    _top_cams(),
            "bot_cams":    _bot_cams(),
        }


@app.post("/capture/top")
def capture_top():
    top = _top_cams()
    with _lock:
        if _state["status"] == Status.CAPTURING:
            raise HTTPException(409, "Capture already in progress")
        _state.update({
            "done_cams": [], "failed_cams": [], "errors": [],
            "logs": [], "images": {},
        })
    _log(">>> /capture/top triggered")
    threading.Thread(target=_run_group, args=(top, "top"), daemon=True).start()
    return {"started": True, "group": "top", "cameras": top}


@app.post("/capture/bottom")
def capture_bottom():
    top = _top_cams()
    bot = _bot_cams()
    with _lock:
        if _state["status"] == Status.CAPTURING:
            raise HTTPException(409, "Capture already in progress")
        missing = [c for c in top if c not in _state["done_cams"]
                   and c not in _state["failed_cams"]]
        if missing:
            raise HTTPException(400, f"Top cameras not yet captured: {missing}")
    _log(">>> /capture/bottom triggered")
    threading.Thread(target=_run_group, args=(bot, "bottom"), daemon=True).start()
    return {"started": True, "group": "bottom", "cameras": bot}


@app.get("/image/{cam_idx}")
def get_image(cam_idx: int):
    with _lock:
        png = _state["images"].get(cam_idx)
    if png is None:
        raise HTTPException(404, f"No image for camera {cam_idx}")
    label = _label_for(cam_idx)
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Cam-Label": label},
    )


@app.get("/scan")
def scan_cameras():
    results = []
    for idx in range(16):
        entry = {"index": idx, "opens": False, "mean": None, "status": "not found"}
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if cap.isOpened():
                entry["opens"] = True
                frame = None
                for _ in range(10):
                    ok, frame = cap.read()
                cap.release()
                if frame is not None:
                    mean = round(float(frame.mean()), 1)
                    entry["mean"] = mean
                    entry["status"] = (
                        "white/overexposed" if mean > 240 else
                        "black/no signal"   if mean < 5  else
                        "OK"                if mean < TARGET_MEAN else
                        f"bright (mean={mean:.0f})"
                    )
                else:
                    entry["status"] = "opens but no frame"
            else:
                entry["status"] = "cannot open"
        except Exception as e:
            entry["status"] = f"error: {e}"
        results.append(entry)
        if idx > 8 and sum(1 for r in results[-3:] if not r["opens"]) == 3:
            break

    found = [r for r in results if r["status"] == "OK"]
    return {
        "cameras": results,
        "working": [r["index"] for r in found],
        "count":   len(found),
    }


@app.post("/inspect")
async def inspect(req: InspectRequest):
    with _lock:
        images = dict(_state["images"])

    if not images:
        raise HTTPException(400, "No images captured yet")

    all_cams = _all_cams()
    missing  = [c for c in all_cams if c not in images]
    if missing:
        raise HTTPException(400, f"Missing cameras: {missing}")

    hf_url = req.hf_url.rstrip("/")
    files  = []
    with _cam_map_lock:
        for entry in _cam_map:
            phys  = entry["physical"]
            label = entry["label"]
            if phys in images:
                files.append(
                    ("images", (f"{label}.png", io.BytesIO(images[phys]), "image/png"))
                )

    data = {"mode": req.mode, "target_sensitivity": str(req.target_sensitivity)}
    _log(f">>> Sending {len(files)} images to {hf_url}/inspect")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{hf_url}/inspect", files=files, data=data)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"HF Space {e.response.status_code}: {e.response.text[:300]}")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Cannot reach HF Space: {e}")

    _log(">>> Inspection complete")
    return JSONResponse({"status": "ok", "html": resp.text})


@app.delete("/session")
def clear_session():
    with _lock:
        count = len(_state["images"])
        _state.update({
            "status": Status.IDLE, "phase": None, "current_cam": None,
            "done_cams": [], "failed_cams": [], "errors": [],
            "logs": [], "images": {},
        })
    _log("Session cleared")
    return {"cleared": count}

# ---------------------------------------------------------------------------
# Serve UI
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).parent

@app.get("/")
def serve_ui():
    f = UI_DIR / "sift.html"
    if not f.exists():
        raise HTTPException(404, "sift.html not found next to jetson_backend.py")
    return FileResponse(str(f), media_type="text/html")

app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    PORT = 8000
    URL  = f"http://localhost:{PORT}"

    print("=" * 55)
    print("  SIFT Jetson Backend v4.0")
    print("=" * 55)
    print(f"  UI: {URL}/")
    print()
    print("  Camera map (edit CAMERA_MAP in this file to change):")
    with _cam_map_lock:
        for e in _cam_map:
            star = " ★" if e.get("is_global") else ""
            print(f"    phys={e['physical']}  →  {e['label']}  [{e['group']:6}]{star}")
    print()
    print(f"  Target mean < {TARGET_MEAN}   Max frames: {MAX_FRAMES}")
    print("  Run configure_cameras.py --scan to find physical indices.")
    print("=" * 55)

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info",
                timeout_graceful_shutdown=1)