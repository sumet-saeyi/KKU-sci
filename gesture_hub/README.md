# Gesture Hub

Real-time hand gesture recognition system using camera input with MediaPipe + YOLOv8 + PyQt6.

## Requirements

```
pip install -r requirements.txt
```

Requires: Python 3.10+, OpenCV, PyQt6, MediaPipe, Ultralytics (YOLOv8), PyTorch

## Run

```bash
python main.py
```

## Project Structure

```
gesture_hub/
├── main.py                    # Entry point – PyQt6 GUI shell
├── core/
│   ├── vision_worker.py       # Camera capture + gesture recognition thread
│   ├── gesture_bus.py         # Signal bus (PyQt signals)
│   ├── gesture_commands.py    # Gesture → command mapping
│   ├── feature_interface.py   # Base class for features
│   └── registry.py            # Feature registry
├── features/                  # Pluggable feature modules
├── gesture_recognizer.task    # MediaPipe gesture model
├── yolov8n.pt                 # YOLOv8 nano model
└── requirements.txt
```

## Notes

### Camera Backend (cross-platform)

`vision_worker.py` automatically selects the camera backend based on the OS:

| OS | Backend | Reason |
|---|---|---|
| **Windows** | `cv2.CAP_DSHOW` (DirectShow) | Prevents MSMF buffer/resolution crashes |
| **Linux** | `cv2.CAP_V4L2` (Video4Linux2) | DirectShow does not exist on Linux and will silently fail to open the camera |

> ⚠️ If the camera still doesn't work, check the following:
> - Camera is connected (`ls /dev/video*` on Linux)
> - No other application is using the camera
> - User has camera access permissions (`sudo usermod -aG video $USER` on Linux)
