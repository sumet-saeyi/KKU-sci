from PyQt6.QtCore import QObject, pyqtSignal

class GestureBus(QObject):
    # Emit frame for UI updates
    frame_ready = pyqtSignal(object) 
    
    # Emit dictionary with gesture data
    gesture_event = pyqtSignal(dict)

bus = GestureBus()
