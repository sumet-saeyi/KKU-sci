"""
Gesture-to-Command mapping system.

MediaPipe GestureRecognizer recognizes these gestures:
  - None, Closed_Fist, Open_Palm, Pointing_Up,
    Thumb_Down, Thumb_Up, Victory, ILoveYou

Map any gesture to a command string. Features subscribe to commands.
To add a new gesture-command, just add one line to GESTURE_MAP.
"""

# ---- Gesture → Command mapping ----
# Edit this dict to add new commands for new features.
GESTURE_MAP = {
    "Pointing_Up": "draw",        # Index finger up → draw
    "Closed_Fist": "stop",        # Fist → stop drawing (keep canvas)
    "Open_Palm":   "clear",       # Open palm → clear canvas
    "Victory":     "color_next",  # Peace sign → next color
    "Thumb_Up":    "predict",     # Thumb up → finish and predict
    "Thumb_Down":  "undo",        # Thumb down → undo
    "ILoveYou":    "color_next",  # ILY sign → next color
    "None":        "idle",        # No gesture detected
}

# All recognized gesture names for reference
ALL_GESTURES = [
    "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
    "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou",
]

def gesture_to_command(gesture_name: str) -> str:
    """Convert a gesture label to a command string."""
    return GESTURE_MAP.get(gesture_name, "idle")
