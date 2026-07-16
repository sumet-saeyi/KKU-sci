import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QHBoxLayout, QListWidget, QStackedWidget,
                             QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

import features.air_canvas
import features.flappy_feature

from core.registry import FEATURE_REGISTRY
from core.vision_worker import VisionWorker
from core.gesture_bus import bus

class GestureHubShell(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Hub")
        self.resize(1000, 700)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        main_layout.addWidget(self.sidebar)
        
        self.content_area = QStackedWidget()
        main_layout.addWidget(self.content_area, 1)
        
        self.camera_view = QLabel("Camera Feed")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet("background-color: black; color: white;")
        self.content_area.addWidget(self.camera_view)
        
        bus.frame_ready.connect(self.update_camera_view)
        
        self.features = []
        for feature_cls in FEATURE_REGISTRY:
            feature_instance = feature_cls()
            self.features.append(feature_instance)
            
            self.sidebar.addItem(f"{feature_instance.icon} {feature_instance.name}")
            
            widget = feature_instance.build_widget()
            self.content_area.addWidget(widget)
            
        self.sidebar.currentRowChanged.connect(self.change_feature)
        
        self.sidebar.insertItem(0, "📷 Camera Raw View")
        self.sidebar.setCurrentRow(0)
        
        self.vision_worker = VisionWorker()
        self.vision_worker.start()
        
    def change_feature(self, index):
        self.content_area.setCurrentIndex(index)
        
    def update_camera_view(self, qt_img):
        if self.content_area.currentIndex() == 0:
            self.camera_view.setPixmap(QPixmap.fromImage(qt_img).scaled(
                self.camera_view.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.vision_worker.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GestureHubShell()
    window.showFullScreen()
    sys.exit(app.exec())
