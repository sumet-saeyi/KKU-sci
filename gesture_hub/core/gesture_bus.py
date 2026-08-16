from PyQt6.QtCore import QObject, pyqtSignal

class GestureBus(QObject):
    # Emit frame for UI updates
    frame_ready = pyqtSignal(object)
    gesture_event = pyqtSignal(dict)

bus = GestureBus()

# Whether VisionWorker restricts hand tracking to the centered detection zone
# and draws the yellow zone box. Only Flappy Bird needs it (keeps players
# from waving outside a small area); other features get the full camera
# capture area. main.py flips this when the active feature changes.
bus.detection_zone_active = False
