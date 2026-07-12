import cv2
import time
import os
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

PROCESS_W = 320
PROCESS_H = 240


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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing MediaPipe model at {model_path}")

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
        # Use DirectShow backend on Windows to prevent MSMF buffer/resolution crashes
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        prev_ts = 0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720)) # Force HD resolution for UI
            display_h, display_w = frame.shape[:2]
            scale_x = display_w / PROCESS_W
            scale_y = display_h / PROCESS_H
            self.frame_count += 1

            # Downscale for processing
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # YOLO person check every 3rd frame
            if self.frame_count % 3 == 0:
                results = self.yolo.predict(
                    small, classes=[0], conf=0.4,
                    verbose=False, imgsz=320)
                self.person_present = len(results[0].boxes) > 0

            # Gesture recognition every 2nd frame
            if self.person_present and self.frame_count % 2 == 0:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_small)
                ts = int(time.time() * 1000)
                if ts <= prev_ts:
                    ts = prev_ts + 1
                prev_ts = ts

                result = self.recognizer.recognize_for_video(mp_image, ts)

                hands = []
                if result.hand_landmarks:
                    self.last_landmarks_raw = result.hand_landmarks
                    for i, hand_lms in enumerate(result.hand_landmarks):
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
