# KKU-Sci Gesture Hub

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyQt6](https://img.shields.io/badge/PyQt-6-green)

## Overview
**Gesture Hub** is an interactive computer vision application built with PyQt6. It leverages **MediaPipe** and **YOLOv8** to track body movements and hand gestures in real-time, allowing users to control applications and games via an augmented reality (AR) "neural uplink" interface.

The project features a sleek, glassmorphism-inspired **Sci-Fi Cybernetic** UI design.

## ⚙️ Tech Stack
- **Python 3.11**
- **PyQt6**: Core UI framework and layout management.
- **Pygame**: Game engine and sprite rendering.
- **MediaPipe (Google)**: Hand tracking, landmark generation, and gesture recognition.
- **YOLOv8 (Ultralytics)**: Fast person-presence gating (GPU-accelerated).
- **OpenCV**: Camera feed capture and processing.
## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sumet-saeyi/KKU-sci.git
   cd KKU-sci/gesture_hub
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Hub**:
   ```bash
   python main.py
   ```

## 🧠 Architecture
- `main.py`: Entry point and main window UI construction.
- `core/vision_worker.py`: Background `QThread` handling OpenCV capture, YOLO bounding boxes, and MediaPipe tracking.
- `core/gesture_bus.py`: Signal-based event bus for broadcasting camera frames and recognized gesture events.
- `features/flappy_feature.py`: The Flappy Bird Pygame hybrid integration.
- `features/spaceship_feature.py`: Spaceship Dodge -- steer with hand X position, Closed_Fist to fire.

## 🕹️ How to Play
1. Stand in front of your webcam.
2. The **Neural Uplink HUD** in the top right will display the active gesture commands.
3. Make the corresponding hand gestures clearly to the camera to control the game!
