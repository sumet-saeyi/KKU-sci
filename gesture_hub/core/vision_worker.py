import cv2
import time
import os
import platform
import urllib.request
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage
from core.gesture_bus import bus
from core.gesture_commands import gesture_to_command

GESTURE_RECOGNIZER_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


def ensure_gesture_model(model_path):
    """gesture_recognizer.task is in .gitignore (model files too large for git),
    so a fresh clone has none. Download Google's official bundle on first run
    instead of hard-crashing with FileNotFoundError."""
    if os.path.exists(model_path):
        return
    print(f"[VisionWorker] Downloading gesture_recognizer.task to {model_path} ...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(GESTURE_RECOGNIZER_URL, model_path)
    print("[VisionWorker] gesture_recognizer.task downloaded.")

# Auto-detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[VisionWorker] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[VisionWorker] GPU: {torch.cuda.get_device_name(0)}")


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

PROCESS_W = 1280
PROCESS_H = 720

# --- Detection Zone Configuration ---
# Whether the zone is enforced is decided per-frame via bus.detection_zone_active
# (main.py sets it based on which feature is currently active -- see gesture_bus.py).
# Zone coordinates as percentages of frame dimensions (0.0 to 1.0)
# A centered box covering 50% of the width and 50% of the height
ZONE_PCT_X1, ZONE_PCT_Y1 = 0.25, 0.25
ZONE_PCT_X2, ZONE_PCT_Y2 = 0.75, 0.75
ZONE_SIZE_PCT = 0.5
# ------------------------------------


def draw_landmarks(image, hand_landmarks, scale_x, scale_y):
    h, w, _ = image.shape
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * PROCESS_W * scale_x)
        y = int(lm.y * PROCESS_H * scale_y)
        points.append((x, y))
    for s, e in HAND_CONNECTIONS:
        cv2.line(image, points[s], points[e], (0, 255, 0), 2)
    for p in points:
        cv2.circle(image, p, 3, (0, 0, 255), -1)


class VisionWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.frame_count = 0

        # YOLO nano on GPU for fast person gating
        self.yolo = YOLO("yolov8n.pt")
        self.yolo.to(DEVICE)

        # MediaPipe Gesture Recognizer (gives landmarks + gesture label)
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'gesture_recognizer.task')
        ensure_gesture_model(model_path)

        # Note: MediaPipe Python GPU Delegate is NOT supported on Windows natively.
        # We must use CPU for MediaPipe. YOLO will still use GPU!
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Cache
        self.last_event = None
        self.last_landmarks_raw = None
        self.person_present = True

    def run(self):
        # Use the appropriate camera backend for the current OS
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_H)
        cap.set(cv2.CAP_PROP_FPS, 60)
        prev_ts = 0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            # Only resize if the camera didn't honor the requested capture
            # size -- avoids an identity resize (same w/h) on every frame.
            if frame.shape[1] != PROCESS_W or frame.shape[0] != PROCESS_H:
                frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            display_h, display_w = frame.shape[:2]
            scale_x = display_w / PROCESS_W
            scale_y = display_h / PROCESS_H
            self.frame_count += 1
            zone_active = bus.detection_zone_active

            # Calculate dynamic zone coordinates based on percentages
            # zx1 = int(display_w * ZONE_PCT_X1)
            # zy1 = int(display_h * ZONE_PCT_Y1)
            # zx2 = int(display_w * ZONE_PCT_X2)
            # zy2 = int(display_h * ZONE_PCT_Y2)
            zone_side = int(min(display_w, display_h) * ZONE_SIZE_PCT)
            zcx, zcy = display_w // 2, display_h // 2
            zx1 = max(0, zcx - zone_side // 2)
            zy1 = max(0, zcy - zone_side // 2)
            zx2 = min(display_w, zcx + zone_side // 2)
            zy2 = min(display_h, zcy + zone_side // 2)

            # frame is already at PROCESS_W x PROCESS_H, so it can be fed to
            # YOLO/MediaPipe directly -- no separate downscale copy needed.
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO person check every 3rd frame -- presence doesn't change
            # fast enough to need checking on every frame.
            if self.frame_count % 3 == 0:
                results = self.yolo.predict(
                    frame, classes=[0], conf=0.4,
                    verbose=False, imgsz=320)
                self.person_present = len(results[0].boxes) > 0

            if self.person_present:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_small)
                ts = int(time.time() * 1000)
                if ts <= prev_ts:
                    ts = prev_ts + 1
                prev_ts = ts

                result = self.recognizer.recognize_for_video(mp_image, ts)

                hands = []
                if result.hand_landmarks:
                    # We won't blindly copy to last_landmarks_raw, we'll only copy valid hands
                    valid_hand_landmarks = []
                    for i, hand_lms in enumerate(result.hand_landmarks):
                        # Filter by zone if enabled
                        if zone_active:
                            # Use middle finger MCP (landmark 9) as the hand center
                            center_x = int(hand_lms[9].x * PROCESS_W * scale_x)
                            center_y = int(hand_lms[9].y * PROCESS_H * scale_y)
                            if not (zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2):
                                continue  # Hand is outside the box zone
                        
                        valid_hand_landmarks.append(hand_lms)
                        draw_landmarks(frame, hand_lms, scale_x, scale_y)
                        
                        gesture_name = "None"
                        if result.gestures and i < len(result.gestures) and len(result.gestures[i]) > 0:
                            gesture_name = result.gestures[i][0].category_name
                            
                        handedness = "Unknown"
                        if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
                            raw_hand = result.handedness[i][0].category_name
                            # The frame is mirrored, so a physical right hand looks like a left hand to MediaPipe
                            if raw_hand == "Left":
                                handedness = "Right"
                            elif raw_hand == "Right":
                                handedness = "Left"
                            
                        cmd = gesture_to_command(gesture_name)
                        
                        lm_list = []
                        for lm in hand_lms:
                            cx = int(lm.x * PROCESS_W * scale_x)
                            cy = int(lm.y * PROCESS_H * scale_y)
                            lm_list.append((cx, cy))
                            
                        hands.append({
                            'gesture': gesture_name,
                            'command': cmd,
                            'handedness': handedness,
                            'landmarks': lm_list
                        })
                    self.last_landmarks_raw = valid_hand_landmarks if valid_hand_landmarks else None
                else:
                    self.last_landmarks_raw = None


                event_data = {
                    'raw_frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'frame_shape': frame.shape,
                    'hands': hands
                }

                self.last_event = event_data
                bus.gesture_event.emit(event_data)

            elif self.last_event is not None:
                # Reuse cached on skipped frames
                self.last_event['raw_frame'] = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB)
                if self.last_landmarks_raw:
                    for hand_lms in self.last_landmarks_raw:
                        draw_landmarks(frame, hand_lms, scale_x, scale_y)
                bus.gesture_event.emit(self.last_event)

            # Draw the detection zone on the frame so the user knows where to put their hand
            if zone_active:
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
                cv2.putText(frame, "Detection Zone", (zx1, zy1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display frame
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_display.shape
            bpl = ch * w
            qt_img = QImage(
                rgb_display.data, w, h, bpl, QImage.Format.Format_RGB888)
            bus.frame_ready.emit(qt_img)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()
