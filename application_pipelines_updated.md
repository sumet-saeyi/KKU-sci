# Native Application Development Pipelines

This document outlines distinct technology stacks and step-by-step pipelines for building standalone native applications using webcam-based gesture control.

## Pipeline 1: Python Desktop Application (Utility App)
*Best for building a background utility app that gives you OS-level control over your computer (like controlling your mouse with your webcam).*

1. **Capture & Display (Camera & UI)**
   - **Tech:** OpenCV + PyQt/Tkinter
   - Use OpenCV to capture the raw webcam video feed and render those frames inside a graphical window.

2. **Extract Coordinates (Tracking)**
   - **Tech:** MediaPipe
   - Pass the raw video frames into the MediaPipe Python library to detect the hand and extract the 3D coordinates of 21 specific hand landmarks.

3. **Process Gestures (Logic)**
   - **Tech:** Python Math
   - Write logic to calculate the Euclidean distance between specific landmarks (e.g., thumb tip [4] and index finger tip [8]). If it drops below a threshold, register a "pinch."

4. **Execute Actions (OS Control)**
   - **Tech:** PyAutoGUI
   - Translate the processed gestures into system-level commands. A pinch triggers `pyautogui.click()`, while velocity mapping triggers `pyautogui.moveTo(x, y)` to drag the cursor.

---

## Pipeline 2: Unity 3D Engine (Visuals & Games)
*Best for highly visual applications, games, XR, or mechanics that rely heavily on physics (like slicing objects Fruit Ninja-style).*

1. **Track & Stream (Camera & Tracking)**
   - **Tech:** MediaPipe Plugins or UDP Python Script
   - Track hands using community-built MediaPipe Unity plugins, or run a lightweight Python script that streams hand coordinates to Unity via a local UDP network connection.

2. **Map to 3D Space (Rigging)**
   - **Tech:** Unity Engine
   - Take incoming 3D coordinates and map them onto a virtual skeleton or invisible collision boxes within your Unity 3D scene.

3. **Enable UI Interaction (Selecting & Clicking)**
   - **Tech:** C# Scripts + Raycasting
   - Attach C# scripts to detect specific poses (like a "fist"). Once detected, use standard Raycasting to select 3D buttons or grab objects.

4. **Apply Collision (Slicing Physics)**
   - **Tech:** Unity Physics
   - Attach a kinematic Rigidbody and collider to the virtual hand. Set collision detection to "Continuous" or "Continuous Speculative" so fast-moving hand slices don't clip through target objects.

---

## Pipeline 3: Python UI-First Architecture (Multithreaded)
*Best practice for Python applications to prevent the heavy video/tracking loop from freezing the Graphical User Interface.*

1. **Initialize the UI Engine**
   - **Tech:** PyQt6 Main Thread
   - Create the main application class using `QMainWindow`. Set up the window layout, buttons, and an empty `QLabel` placeholder for the live video feed.

2. **Build the Vision Worker**
   - **Tech:** QThread Background Worker
   - Create a separate `QThread` class. Inside this worker thread, initialize the OpenCV webcam capture (`cv2.VideoCapture`) and MediaPipe tracking model.

3. **Process and Route Data**
   - **Tech:** Cross-Thread Communication
   - Start a `while` loop in the worker to read frames and extract landmarks. Convert frames to PyQt-friendly format (`QImage` or `QPixmap`) and use PyQt Signals to broadcast the image safely.

4. **Connect Signals to the Display**
   - **Tech:** UI Update
   - In the Main UI Thread, connect the worker's image Signal to a function (Slot). Every time a new frame is emitted, update the `QLabel` to show a smooth, live video feed without UI freezing.

5. **Calculate Gestures and Execute**
   - **Tech:** Math + PyAutoGUI
   - While processing frames in the worker thread, calculate landmark distances. When a gesture is recognized, emit an action Signal to trigger `pyautogui.click()` or `pyautogui.moveTo()` to control the OS safely.

---

## Pipeline 4: Custom Application Concept — Gesture Hub (Extensible Multi-Feature Shell)
*Updated per your note: the UI engine should make it easy to add new features or windows later. Air Canvas is now Feature Module #1 inside an extensible shell, not the whole app. Say if that's off.*

**Core Concept:** A gesture-controlled desktop app built as an extensible shell rather than one fixed tool. Each capability — drawing/annotation, mouse control, a presentation remote, whatever comes next — is a self-contained "feature module" that plugs into the same shell. Adding a new feature or a new window later means writing one new module, not reworking the app.

**Custom Pipeline Steps:**

1. **Multithreaded Core (from Pipeline 3)**
   - **Tech:** PyQt6 + QThread
   - Keep the camera/tracking loop on a background thread exactly as in Pipeline 3. This layer is the foundation and never changes as features get added on top.

2. **Central Gesture Bus (Decoupling)**
   - **Tech:** PyQt Signals as a Pub/Sub Bus
   - Instead of the tracking thread calling feature code directly, have it emit landmark/gesture events onto one shared `GestureBus` signal. Features subscribe to this bus; the core thread never has to know which features exist.

3. **Feature Module Interface**
   - **Tech:** Python Abstract Base Class
   - Define one interface every feature implements — `name`, `icon`, `build_widget()`, `on_gesture(event)`. A new feature is just a new class implementing this interface; nothing in the core app has to change to support it.

4. **Feature Registry + Launcher**
   - **Tech:** Python List/Dict Registry + Sidebar UI
   - Keep one list of registered feature classes. The main window reads this list to auto-build a sidebar or launcher menu, so registering a feature (one line of code) is what makes it appear in the UI.

5. **Window Manager (Tabs or Standalone)**
   - **Tech:** `QStackedWidget` for tabs, `QMainWindow` per feature for standalone
   - Give each feature a flag: `embedded` (opens as a page inside the main shell) or `standalone` (opens in its own floating window). One small WindowManager class spawns, tracks, and closes these by key, so new windows never need one-off boilerplate.

6. **First Feature Module: Air Canvas**
   - **Tech:** NumPy Canvas + Gesture Vocabulary (as designed earlier)
   - Implement the earlier drawing/annotation idea as Feature Module #1 — persistent NumPy canvas, pinch-to-pick-color, palm-to-clear — to prove the plug-in pattern works before adding a second feature.
