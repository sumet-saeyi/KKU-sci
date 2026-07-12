import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = hand_landmarks[start_idx]
        p2 = hand_landmarks[end_idx]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

class VisionWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        
        # Initialize Tasks API
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process with timestamp in ms
            timestamp_ms = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(rgb_frame, hand_landmarks)
                    
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.change_pixmap_signal.emit(qt_img)
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline 3: Base UI App (Tasks API)")
        self.resize(800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.video_label = QLabel("Starting webcam...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.video_label)
        
        self.worker = VisionWorker()
        self.worker.change_pixmap_signal.connect(self.update_image)
        self.worker.start()
        
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
