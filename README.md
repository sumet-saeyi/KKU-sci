# KKU-Sci Gesture Hub

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyQt6](https://img.shields.io/badge/PyQt-6-green) ![Docker](https://img.shields.io/badge/Docker-Supported-blue)

## Overview
**Gesture Hub** is an interactive computer vision application built with PyQt6. It leverages **MediaPipe** and **YOLOv8** to track body movements and hand gestures in real-time, allowing users to control applications and games via an augmented reality (AR) "neural uplink" interface.

The project features a sleek, glassmorphism-inspired **Sci-Fi Cybernetic** UI design.

## 🎮 Features
- **Flappy Bird AR Integration**: Play the classic arcade game entirely through hand gestures!
  - ☝️ **Point Up (Index Finger)**: Flap / Jump
  - 🖐️ **Open Palm**: Restart Game
- **Sci-Fi Cybernetic UI**: Semi-transparent dark glass components, glowing Teal (`#0ea5e9`) and Amber (`#f59e0b`) accents, and a tinted cyber-simulation game environment.
- **Dynamic Cyber Leaderboard**: A rich HTML-rendered persistent scoreboard that automatically assigns unique `Anonymous_N` IDs if no name is provided.
- **60 FPS Physics Engine**: Rebuilt Pygame floating-point mechanics for buttery smooth gameplay synchronized with the AR overlay.

## ⚙️ Tech Stack
- **Python 3.11**
- **PyQt6**: Core UI framework and layout management.
- **Pygame**: Game engine and sprite rendering.
- **MediaPipe (Google)**: Hand tracking, landmark generation, and gesture recognition.
- **YOLOv8 (Ultralytics)**: Fast person-presence gating (GPU-accelerated).
- **OpenCV**: Camera feed capture and processing.
- **PyInstaller**: Standalone Windows Executable generation.
- **Docker**: Containerized Linux deployment support.

## 🚀 Installation & Setup

### Option 1: Native Python (Windows/Linux/Mac)
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

### Option 2: Docker (Linux)
The repository includes a `Dockerfile` pre-configured with the heavy X11, OpenCV, and Pygame Audio dependencies required to run the UI.
```bash
docker build -t gesture_hub .
# Note: Running GUI apps with webcam access via Docker requires passing X11 sockets and /dev/video devices.
```

## 📦 Standalone Executable (Windows)
You can build a portable Windows `.exe` using PyInstaller. 
Run the following command inside the `gesture_hub` folder:
```bash
pyinstaller --name "GestureHub" --add-data "features/flappy_assets;features/flappy_assets" --add-data "gesture_recognizer.task;." --add-data "yolov8n.pt;." main.py
```
This will output a `dist/GestureHub` folder containing the executable and all bundled AI models. You can share this folder, and it will run instantly without requiring Python to be installed!

## 🧠 Architecture
- `main.py`: Entry point and main window UI construction.
- `core/vision_worker.py`: Background `QThread` handling OpenCV capture, YOLO bounding boxes, and MediaPipe tracking.
- `core/gesture_bus.py`: Signal-based event bus for broadcasting camera frames and recognized gesture events.
- `features/flappy_feature.py`: The Flappy Bird Pygame hybrid integration.

## 🕹️ How to Play
1. Stand in front of your webcam.
2. The **Neural Uplink HUD** in the top right will display the active gesture commands.
3. Make the corresponding hand gestures clearly to the camera to control the game!
