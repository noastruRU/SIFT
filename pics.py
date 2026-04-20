import cv2
import time
import os
from datetime import datetime

NUM_CAMERAS = 9
BASE_DIR = "camera_images"


os.makedirs(BASE_DIR, exist_ok=True)

WARMUP_FRAMES = 15
FRAME_DELAY = 0.07

# Create a unique folder for this round
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
round_dir = os.path.join(BASE_DIR, f"FOR_NOA_{timestamp}")
os.makedirs(round_dir, exist_ok=True)

for cam_id in range(NUM_CAMERAS):
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"Camera {cam_id} could not be opened")
        continue

    time.sleep(0.5)

    # Warm-up frames
    for _ in range(WARMUP_FRAMES):
        cap.read()
        time.sleep(FRAME_DELAY)

    ret, frame = cap.read()

    if ret:
        filename = os.path.join(round_dir, f"cam_{cam_id}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
    else:
        print(f"Failed to capture from camera {cam_id}")

    cap.release()

print(f"Done. Images saved in {round_dir}")