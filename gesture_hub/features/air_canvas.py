import numpy as np
import cv2
import time
import os
import json
import subprocess
import torch
from torchvision import transforms
from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from core.feature_interface import FeatureModule
from core.registry import register_feature
from core.gesture_bus import bus
from train_mnist import Ge2019CNN

# Color palette for cycling (Black and White mode)
COLORS = [
    (255, 255, 255) # Pure White
]


@register_feature
class AirCanvasFeature(FeatureModule):
    def __init__(self):
        self.canvas = None
        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.display_label = QLabel("Air Canvas")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.display_label)
        self.widget.setLayout(self.layout)

        self.color_index = 0
        self.drawing_color = COLORS[0]
        self.brush_size = 5
        
        # Track state for multiple hands using their "handedness" label
        self.px = {'Left': 0, 'Right': 0, 'Unknown': 0}
        self.py = {'Left': 0, 'Right': 0, 'Unknown': 0}

        # Smoothing
        self.smooth_buffer = {
            'Left': deque(maxlen=5),
            'Right': deque(maxlen=5),
            'Unknown': deque(maxlen=5)
        }

        # Gesture stability
        self.command_history = {
            'Left': deque(maxlen=8),
            'Right': deque(maxlen=8),
            'Unknown': deque(maxlen=8)
        }
        self.current_command = {'Left': 'idle', 'Right': 'idle', 'Unknown': 'idle'}

        # Undo stack
        self.undo_stack = []

        # MNIST ML setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mnist_model = None
        self.mlp_path = os.path.join(os.path.dirname(__file__), "..", "mnist_mlp.pth")
        
        # Prediction Animation State
        self.prediction_result = None
        self.prediction_timer = 0
        self.digit_bbox = None
        self.processed_digit_img = None

        bus.gesture_event.connect(self.on_gesture)

    @property
    def name(self) -> str:
        return "Air Canvas"

    @property
    def icon(self) -> str:
        return "🖌️"

    def build_widget(self) -> QWidget:
        return self.widget

    def _smooth(self, h_id, x, y):
        if len(self.smooth_buffer[h_id]) == 0:
            self.smooth_buffer[h_id].append((x, y))
            return (x, y)
            
        last_x, last_y = self.smooth_buffer[h_id][-1]
        
        # Calculate distance moved
        dist = ((last_x - x)**2 + (last_y - y)**2)**0.5
        
        # Deadzone: ignore micro-jitters
        if dist < 3.0:
            return (last_x, last_y)
            
        # Dynamic alpha: fast movements track closer, slow movements are heavily smoothed
        alpha = min(0.4, max(0.08, dist / 150.0))
        
        new_x = int(last_x * (1 - alpha) + x * alpha)
        new_y = int(last_y * (1 - alpha) + y * alpha)
        
        self.smooth_buffer[h_id].append((new_x, new_y))
        return (new_x, new_y)

    def _stable_command(self, h_id, cmd):
        self.command_history[h_id].append(cmd)
        counts = {}
        for c in self.command_history[h_id]:
            counts[c] = counts.get(c, 0) + 1
        for c, n in counts.items():
            if n >= 5:
                self.current_command[h_id] = c
                return c
        return self.current_command[h_id]

    def on_gesture(self, event_data: dict):
        if not self.widget.isVisible():
            return

        raw_frame = event_data['raw_frame']
        if self.canvas is None or self.canvas.shape != raw_frame.shape:
            self.canvas = np.zeros_like(raw_frame)

        hands = event_data.get('hands', [])
        
        # Track active commands for HUD
        active_commands = []
        
        # First pass: update stable commands for all detected hands
        hand_commands = {}
        for hand in hands:
            command = hand.get('command', 'idle')
            h_id = hand.get('handedness', 'Unknown')
            landmarks = hand.get('landmarks', [])
            stable = self._stable_command(h_id, command)
            hand_commands[h_id] = (stable, landmarks)
            active_commands.append(f"{h_id}: {stable.upper()}")
            
        # Global multi-hand commands
        both_clear = False
        if 'Left' in hand_commands and 'Right' in hand_commands:
            if hand_commands['Left'][0] == 'clear' and hand_commands['Right'][0] == 'clear':
                both_clear = True
                
        # Left Hand Clutch: If Left hand is a fist ('stop'), block the Right hand from drawing
        left_stable = hand_commands.get('Left', ('idle', []))[0]
        drawing_enabled = (left_stable != 'stop')
                
        if both_clear:
            # Open palm on BOTH hands → clear canvas and cancel prediction animations
            self.undo_stack.append(self.canvas.copy())
            self.canvas = np.zeros_like(raw_frame)
            for h_id in ['Left', 'Right', 'Unknown']:
                self.px[h_id], self.py[h_id] = 0, 0
                self.smooth_buffer[h_id].clear()
                
            # Cancel animation!
            self.prediction_result = None
            self.anim_frame = 0

        # Execute individual hand commands
        for h_id, (stable, landmarks) in hand_commands.items():
            if stable == "clear":
                # Ignored individually, only triggers when both_clear is True
                pass

            elif stable == "draw" and landmarks:
                if h_id == "Right" and drawing_enabled:
                    # Index finger → draw
                    sx, sy = self._smooth(h_id, landmarks[8][0], landmarks[8][1])
                    if self.px[h_id] == 0 and self.py[h_id] == 0:
                        self.px[h_id], self.py[h_id] = sx, sy
                    cv2.line(self.canvas, (self.px[h_id], self.py[h_id]), (sx, sy),
                             self.drawing_color, self.brush_size)
                    self.px[h_id], self.py[h_id] = sx, sy
                else:
                    self.px[h_id], self.py[h_id] = 0, 0
                    self.smooth_buffer[h_id].clear()

            elif stable == "color_next":
                # Victory/peace → next color
                self.color_index = (self.color_index + 1) % len(COLORS)
                self.drawing_color = COLORS[self.color_index]
                self.px[h_id], self.py[h_id] = 0, 0
                self.smooth_buffer[h_id].clear()

            elif stable == "brush_up":
                self.brush_size = min(self.brush_size + 1, 20)
                self.px[h_id], self.py[h_id] = 0, 0

            elif stable == "brush_down":
                self.brush_size = max(self.brush_size - 1, 2)
                self.px[h_id], self.py[h_id] = 0, 0

            elif stable == "predict":
                # Thumbs Up → Predict digit using MLP
                if getattr(self, 'prediction_result', None) is None:
                    self._run_prediction()
                self.px[h_id], self.py[h_id] = 0, 0

            elif stable == "undo":
                if self.undo_stack:
                    self.canvas = self.undo_stack.pop()
                self.px[h_id], self.py[h_id] = 0, 0

            elif stable == "stop":
                # Fist → stop drawing, keep canvas
                self.px[h_id], self.py[h_id] = 0, 0
                self.smooth_buffer[h_id].clear()

            else:
                self.px[h_id], self.py[h_id] = 0, 0
                self.smooth_buffer[h_id].clear()

        # ---- HUD overlay ----
        indicator = raw_frame.copy()
        
        # Display static usage hints instead of active hand state
        hints = [
            "Draw: Pointing Up (Right)",
            "Lock: Closed Fist (Left)",
            "Predict: Thumb Up",
            "Clear All: Two Open Palms",
            "Undo: Thumb Down",
            "Stop Drawing: Closed Fist (Right)"
        ]
        y_offset = 30
        for hint in hints:
            cv2.putText(indicator, hint, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            y_offset += 30

        # Brush indicator
        cv2.circle(indicator, (580, 40), self.brush_size,
                   self.drawing_color, -1)
        cv2.putText(indicator, f"Brush: {self.brush_size}", (550, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        img = cv2.addWeighted(indicator, 0.5, self.canvas, 0.5, 0)
        
        # Overlay Real-time OpenCV Network Animation
        if getattr(self, 'prediction_result', None) is not None:
            img = self._draw_dynamic_network(img)
            
        h, w, ch = img.shape
        bpl = ch * w
        qt_img = QImage(img.data, w, h, bpl, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _draw_dynamic_network(self, base_img):
        """ Draws a clean, dashboard-style CNN + MLP forward-pass HUD.

        Visual language: dark slate background, muted panel borders, a
        teal/amber accent palette, and a top "pipeline stepper" showing
        overall progress -- closer to a product analytics dashboard than a
        hacker-terminal effect.

        The underlying animation logic still matches how the network
        actually computes (kept from the previous accuracy pass):
          - Conv layers: a small receptive-field window scans in raster
            order; the same relative position is revealed simultaneously
            across every output channel (weights are shared spatially and
            all channels are computed in parallel).
          - Pooling: every non-overlapping window resolves at the same
            instant, shown as a synchronized highlight, not a global squish.
          - Flatten: values are reshaped into a vector -- drawn as a ribbon
            unrolling to full length, nothing funneled or lost.
          - Dense (FC) layers: every neuron is computed in parallel from the
            same input vector, so a layer lights up in one synchronized
            flash, never a traveling wave.
        """
        if not hasattr(self, 'anim_frame'):
            self.anim_frame = 0
        self.anim_frame += 1
        f = self.anim_frame

        # ---------------------------------------------------------------
        # Palette (Slate dashboard theme, BGR tuples for OpenCV)
        # ---------------------------------------------------------------
        BG        = (36, 26, 18)     # slate-900
        PANEL_BG  = (46, 35, 26)     # slightly lighter panel fill
        BORDER    = (85, 65, 51)     # slate-700
        BORDER_DIM = (58, 46, 38)    # dimmer border for pending stages
        GRID      = (48, 38, 30)     # faint background grid
        TEXT      = (240, 232, 226)  # slate-200
        TEXT_DIM  = (184, 163, 148)  # slate-400
        TEAL      = (191, 212, 45)   # teal-400 accent (active/flow)
        TEAL_DIM  = (120, 135, 40)   # dimmed teal (completed)
        AMBER     = (36, 191, 251)   # amber-400 (winner / highlight)

        def dim(color, factor):
            return tuple(int(c * factor) for c in color)

        # ---------------------------------------------------------------
        # Background
        # ---------------------------------------------------------------
        overlay = base_img.copy()
        overlay[:] = BG
        for x in range(0, overlay.shape[1], 60):
            cv2.line(overlay, (x, 0), (x, overlay.shape[0]), GRID, 1)
        for y in range(0, overlay.shape[0], 60):
            cv2.line(overlay, (0, y), (overlay.shape[1], y), GRID, 1)
        img = cv2.addWeighted(overlay, 0.94, base_img, 0.06, 0)
        h, w = img.shape[:2]

        def panel(x, y, pw, ph, border=BORDER, fill=PANEL_BG, accent=None):
            cv2.rectangle(img, (x, y), (x + pw, y + ph), fill, -1)
            cv2.rectangle(img, (x, y), (x + pw, y + ph), border, 1)
            if accent is not None:
                cv2.line(img, (x, y), (x + pw, y), accent, 2)

        def title(text, x, y, color=TEXT_DIM, scale=0.42):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        def clip01(t):
            return max(0.0, min(1.0, t))

        # ---------------------------------------------------------------
        # Pipeline stage schedule (single source of truth, used by both
        # the top stepper and the panel accents below)
        # ---------------------------------------------------------------
        STAGES = [
            ("INPUT",   0,   30),
            ("CONV1",   30,  94),
            ("POOL1",   94,  108),
            ("CONV2",   108, 172),
            ("POOL2",   172, 186),
            ("FLATTEN", 186, 200),
            ("FC1",     200, 214),
            ("FC2",     214, 228),
            ("OUTPUT",  228, 258),
        ]

        def stage_state(i):
            _, s, e = STAGES[i]
            if f < s:
                return 'pending'
            if f < e:
                return 'active'
            return 'done'

        # ---------------------------------------------------------------
        # Header: title + pipeline stepper
        # ---------------------------------------------------------------
        title("GE-2019  /  FORWARD PASS", 40, 30, color=TEXT, scale=0.55)
        cv2.line(img, (40, 42), (400, 42), BORDER, 1)

        step_y = 70
        step_x0, step_x1 = 40, w - 40
        n = len(STAGES)
        xs = [int(step_x0 + (step_x1 - step_x0) * i / (n - 1)) for i in range(n)]
        for i in range(n - 1):
            done_next = stage_state(i) == 'done'
            line_col = TEAL_DIM if done_next else BORDER_DIM
            cv2.line(img, (xs[i], step_y), (xs[i + 1], step_y), line_col, 2)
        for i, (name, s, e) in enumerate(STAGES):
            st = stage_state(i)
            if st == 'pending':
                cv2.circle(img, (xs[i], step_y), 5, BORDER_DIM, 1)
                col = TEXT_DIM
            elif st == 'active':
                pulse = 0.6 + 0.4 * np.sin(f * 0.3)
                cv2.circle(img, (xs[i], step_y), 6, dim(TEAL, pulse), -1)
                cv2.circle(img, (xs[i], step_y), 8, TEAL, 1)
                col = TEXT
            else:
                cv2.circle(img, (xs[i], step_y), 5, TEAL_DIM, -1)
                col = TEXT_DIM
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(img, name, (xs[i] - tw // 2, step_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        def local_raster_mask(H, W, cell_h, cell_w, sub_blocks, progress):
            """True for pixels whose position-within-its-cell has been
            'computed' yet, in raster order, at the SAME relative offset in
            every cell simultaneously -- lets every output channel fill in
            together instead of one-channel/one-column-at-a-time."""
            yy, xx = np.mgrid[0:H, 0:W]
            local_y = yy % cell_h
            local_x = xx % cell_w
            sub_h = max(1, cell_h // sub_blocks)
            sub_w = max(1, cell_w // sub_blocks)
            sub_row = np.minimum(local_y // sub_h, sub_blocks - 1)
            sub_col = np.minimum(local_x // sub_w, sub_blocks - 1)
            idx = sub_row * sub_blocks + sub_col
            thresh = progress * (sub_blocks * sub_blocks)
            return idx < thresh

        def scan_rowcol(sub_blocks, progress):
            idx = int(clip01(progress) * (sub_blocks * sub_blocks))
            idx = min(idx, sub_blocks * sub_blocks - 1)
            return idx // sub_blocks, idx % sub_blocks

        def burst(from_box, to_pt, offset):
            pass

        def draw_bezier(img, p0, p3, w_val, alpha_mul=1.0):
            if alpha_mul <= 0: return
            abs_w = min(1.0, abs(w_val) * 1.5)
            if abs_w < 0.1: return
            
            c = dim(TEAL, abs_w * 0.4 * alpha_mul) if w_val > 0 else dim(AMBER, abs_w * 0.4 * alpha_mul)
            
            pts = []
            dx = p3[0] - p0[0]
            p1 = (p0[0] + dx // 2, p0[1])
            p2 = (p3[0] - dx // 2, p3[1])
            for t in np.linspace(0, 1, 10):
                t2, mt = t * t, 1 - t
                mt2 = mt * mt
                x = int((mt2*mt)*p0[0] + 3*(mt2)*t*p1[0] + 3*mt*(t2)*p2[0] + (t2*t)*p3[0])
                y = int((mt2*mt)*p0[1] + 3*(mt2)*t*p1[1] + 3*mt*(t2)*p2[1] + (t2*t)*p3[1])
                pts.append([x, y])
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, c, 1, cv2.LINE_AA)

        SUB = 8
        in_x, in_y, in_s = 20, 260, 180
        c1_x, c1_y, c1_w, c1_h = 240, 280, 288, 144
        c2_x, c2_y, c2_s = 570, 260, 176
        fc1_x = 830
        fc2_x = 930
        out_x = 1030
        mid_y = 350

        panel(in_x - 12, in_y - 34, in_s + 24, in_s + 56, accent=TEAL if stage_state(0) != 'pending' else None)
        title("INPUT  28x28", in_x, in_y - 14, color=TEXT_DIM)

        if hasattr(self, 'network_input') and f > 10:
            disp_in = cv2.cvtColor(self.network_input, cv2.COLOR_GRAY2BGR)
            disp_in = cv2.resize(disp_in, (in_s, in_s), interpolation=cv2.INTER_NEAREST)
            if 10 < f < 30:
                glow = np.sin((f - 10) / 20.0 * np.pi)
                disp_in = cv2.addWeighted(disp_in, 1.0, np.full_like(disp_in, int(60 * glow)), 1.0, 0)
            img[in_y:in_y+in_s, in_x:in_x+in_s] = disp_in
            cv2.rectangle(img, (in_x, in_y), (in_x + in_s, in_y + in_s), BORDER, 1)

        conv1_start, conv1_end = 30, 94
        p1 = clip01((f - conv1_start) / (conv1_end - conv1_start))
        st1 = stage_state(1)
        panel(c1_x - 12, c1_y - 34, c1_w + 24, c1_h + 56,
              accent=(TEAL if st1 == 'active' else (TEAL_DIM if st1 == 'done' else None)))
        title("CONV1   32 channels, shared kernel", c1_x, c1_y - 14, color=(TEXT if st1 != 'pending' else TEXT_DIM))

        if hasattr(self, 'hud_c1_act') and f > conv1_start:
            act1 = self.hud_c1_act
            cell_h, cell_w = c1_h // 4, c1_w // 8
            mask = local_raster_mask(c1_h, c1_w, cell_h, cell_w, SUB, p1)
            act1_show = np.zeros_like(act1)
            act1_show[mask] = act1[mask]
            img[c1_y:c1_y+c1_h, c1_x:c1_x+c1_w] = act1_show
            cv2.rectangle(img, (c1_x, c1_y), (c1_x + c1_w, c1_y + c1_h), BORDER, 1)
            for gx in range(c1_x, c1_x + c1_w, cell_w):
                cv2.line(img, (gx, c1_y), (gx, c1_y + c1_h), BG, 1)
            for gy in range(c1_y, c1_y + c1_h, cell_h):
                cv2.line(img, (c1_x, gy), (c1_x + c1_w, gy), BG, 1)

        if conv1_start < f < conv1_end:
            row, col = scan_rowcol(SUB, p1)
            bw, bh = in_s // SUB, in_s // SUB
            bx, by = in_x + col*bw, in_y + row*bh
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), TEAL, 2)
            title("kernel", in_x, in_y + in_s + 16, color=TEAL, scale=0.36)
            cell_h, cell_w = c1_h // 4, c1_w // 8
            for (cr, cc) in [(0, 0), (1, 3), (2, 5), (3, 7)]:
                tx = c1_x + cc*cell_w + int(col/SUB*cell_w)
                ty = c1_y + cr*cell_h + int(row/SUB*cell_h)
                cv2.line(img, (bx+bw, by+bh//2), (tx, ty), dim(TEAL, 0.35), 1, cv2.LINE_AA)

        pool1_start, pool1_end = 94, 108
        if pool1_start < f < pool1_end and hasattr(self, 'hud_c1_act'):
            pulse = 0.5 + 0.4 * np.sin(f * 0.6)
            step = 20
            gcol = dim(TEAL, pulse)
            for gy in range(c1_y, c1_y + c1_h, step):
                cv2.line(img, (c1_x, gy), (c1_x + c1_w, gy), gcol, 1)
            for gx in range(c1_x, c1_x + c1_w, step):
                cv2.line(img, (gx, c1_y), (gx, c1_y + c1_h), gcol, 1)
            title("POOL1  2x2 max, resolved together", c1_x, c1_y + c1_h + 18, color=TEAL, scale=0.36)

        if f >= pool1_start:
            burst((c1_x, c1_y, c1_w, c1_h), (c2_x - 10, c2_y + c2_s//2), f - pool1_end)

        conv2_start, conv2_end = 108, 172
        p2 = clip01((f - conv2_start) / (conv2_end - conv2_start))
        st2 = stage_state(3)
        panel(c2_x - 12, c2_y - 34, c2_s + 24, c2_s + 56,
              accent=(TEAL if st2 == 'active' else (TEAL_DIM if st2 == 'done' else None)))
        title("CONV2   64 channels, shared kernel", c2_x, c2_y - 14, color=(TEXT if st2 != 'pending' else TEXT_DIM))

        if hasattr(self, 'hud_c2_act') and f > conv2_start:
            act2 = self.hud_c2_act
            cell = c2_s // 8
            mask = local_raster_mask(c2_s, c2_s, cell, cell, SUB, p2)
            act2_show = np.zeros_like(act2)
            act2_show[mask] = act2[mask]
            img[c2_y:c2_y+c2_s, c2_x:c2_x+c2_s] = act2_show
            cv2.rectangle(img, (c2_x, c2_y), (c2_x + c2_s, c2_y + c2_s), BORDER, 1)
            for gx in range(c2_x, c2_x + c2_s, cell):
                cv2.line(img, (gx, c2_y), (gx, c2_y + c2_s), BG, 1)
            for gy in range(c2_y, c2_y + c2_s, cell):
                cv2.line(img, (c2_x, gy), (c2_x + c2_s, gy), BG, 1)

        if conv2_start < f < conv2_end:
            row, col = scan_rowcol(SUB, p2)
            bw, bh = c1_w // SUB, c1_h // SUB
            bx, by = c1_x + col*bw, c1_y + row*bh
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), TEAL, 2)
            cell = c2_s // 8
            for (cr, cc) in [(0, 0), (2, 3), (5, 5), (7, 7)]:
                tx = c2_x + cc*cell + int(col/SUB*cell)
                ty = c2_y + cr*cell + int(row/SUB*cell)
                cv2.line(img, (bx+bw, by+bh//2), (tx, ty), dim(TEAL, 0.35), 1, cv2.LINE_AA)

        pool2_start, pool2_end = 172, 186
        if pool2_start < f < pool2_end and hasattr(self, 'hud_c2_act'):
            pulse = 0.5 + 0.4 * np.sin(f * 0.6)
            step = 20
            gcol = dim(TEAL, pulse)
            for gy in range(c2_y, c2_y + c2_s, step):
                cv2.line(img, (c2_x, gy), (c2_x + c2_s, gy), gcol, 1)
            for gx in range(c2_x, c2_x + c2_s, step):
                cv2.line(img, (gx, c2_y), (gx, c2_y + c2_s), gcol, 1)
            title("POOL2  2x2 max, resolved together", c2_x, c2_y + c2_s + 18, color=TEAL, scale=0.36)

        flat_start, flat_end = 186, 200
        ft = clip01((f - flat_start) / (flat_end - flat_start))
        ribbon_x0 = c2_x + c2_s + 15
        ribbon_x1 = fc1_x - 15
        if f > flat_start:
            length = int((ribbon_x1 - ribbon_x0) * ft)
            cv2.line(img, (ribbon_x0, mid_y), (ribbon_x0 + max(length, 1), mid_y), TEAL, 1, cv2.LINE_AA)
            for i in range(0, length, 5):
                cv2.line(img, (ribbon_x0 + i, mid_y - 5), (ribbon_x0 + i, mid_y + 5), dim(TEAL, 0.6), 1)
            title("FLATTEN", ribbon_x0, mid_y - 16, color=TEAL, scale=0.34)

        fc1_start = 200
        st_fc1 = stage_state(6)
        panel(fc1_x - 20, 216, 40, 264,
              accent=(TEAL if st_fc1 == 'active' else (TEAL_DIM if st_fc1 == 'done' else None)))
        if f > fc1_start:
            onset = f - fc1_start
            bright = onset < 8
            title("FC1", fc1_x - 12, 208, color=(TEXT if bright or st_fc1 != 'pending' else TEXT_DIM))
            for i in range(30):
                y = 230 + i * 8
                alpha = min(1.0, onset / 10.0)
                for jy in range(mid_y - 20 + (i % 4) * 4, mid_y + 20, 16):
                    weight = (hash((i, jy)) % 200 - 100) / 100.0
                    draw_bezier(img, (ribbon_x1, jy), (fc1_x, y), weight, alpha)
                cv2.circle(img, (fc1_x, y), 3, TEAL if bright else TEAL_DIM, -1)
            title("dense weights activated", fc1_x - 30, 490, color=TEXT_DIM, scale=0.34)

        burst((fc1_x - 5, 225, 10, 240), (fc2_x - 5, 350), f - 210)

        fc2_start = 214
        st_fc2 = stage_state(7)
        panel(fc2_x - 20, 236, 40, 224,
              accent=(TEAL if st_fc2 == 'active' else (TEAL_DIM if st_fc2 == 'done' else None)))
        if f > fc2_start:
            onset = f - fc2_start
            bright = onset < 8
            title("FC2", fc2_x - 12, 228, color=(TEXT if bright or st_fc2 != 'pending' else TEXT_DIM))
            for i in range(25):
                y2 = 250 + i * 8
                alpha = min(1.0, onset / 10.0)
                for j in range(i % 12, 30, 12):
                    y1 = 230 + j * 8
                    weight = (hash((i, j, "fc2")) % 200 - 100) / 100.0
                    draw_bezier(img, (fc1_x, y1), (fc2_x, y2), weight, alpha)
                cv2.circle(img, (fc2_x, y2), 3, TEAL if bright else TEAL_DIM, -1)

        burst((fc2_x - 5, 245, 10, 200), (out_x - 5, 350), f - 226)

        out_start = 228
        st_out = stage_state(8)
        panel(out_x - 12, 228, 190, 216,
              accent=(TEAL if st_out == 'active' else (TEAL_DIM if st_out == 'done' else None)))
        title("OUTPUT  softmax", out_x, 220, color=(TEXT if st_out != 'pending' else TEXT_DIM), scale=0.36)

        for j in range(10):
            oy = 260 + j * 18
            if f > out_start:
                onset = f - out_start
                alpha = min(1.0, onset / 10.0)
                for i in range(j % 8, 25, 8):
                    y2 = 250 + i * 8
                    weight = (hash((i, j, "out")) % 200 - 100) / 100.0
                    draw_bezier(img, (fc2_x, y2), (out_x, oy), weight, alpha)
            if hasattr(self, 'hud_probs') and f > out_start:
                grow_t = min(1.0, (f - out_start) / 20.0)
                prob = self.hud_probs[j] * grow_t
            else:
                prob = 0.0

            is_winner = f > out_start and getattr(self, 'prediction_result', -1) == j
            track_col = BORDER_DIM
            fill_col = AMBER if is_winner else TEAL_DIM
            text_col = TEXT if is_winner else TEXT_DIM

            cv2.rectangle(img, (out_x + 16, oy - 6), (out_x + 16 + 110, oy + 6), track_col, 1)
            cv2.rectangle(img, (out_x + 16, oy - 6), (out_x + 16 + int(110 * prob), oy + 6), fill_col, -1)
            cv2.putText(img, str(j), (out_x, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_col, 1, cv2.LINE_AA)
            cv2.putText(img, f"{prob:.2f}", (out_x + 134, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_col, 1, cv2.LINE_AA)

            if is_winner and f > out_start + 20:
                pass

        # ---------------------------------------------------------------
        # Result card
        # ---------------------------------------------------------------
        if f > out_start + 30:
            card_w, card_h = 420, 60
            cx, cy = w // 2 - card_w // 2, h - 80
            panel(cx, cy, card_w, card_h, border=BORDER, fill=PANEL_BG, accent=AMBER)
            cv2.putText(img, "PREDICTION", (cx + 20, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(img, str(self.prediction_result), (cx + 20, cy + 52), cv2.FONT_HERSHEY_DUPLEX, 1.1, AMBER, 2, cv2.LINE_AA)
            conf = float(np.max(self.hud_probs)) if hasattr(self, 'hud_probs') else 0.0
            cv2.putText(img, f"confidence {conf*100:.1f}%", (cx + 70, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)

        return img

    def _run_prediction(self):
        # 1. Load model if needed
        if self.mnist_model is None:
            if not os.path.exists(self.mlp_path):
                print("[AirCanvas] Model not trained yet!")
                return
            self.mnist_model = Ge2019CNN().to(self.device)
            self.mnist_model.load_state_dict(torch.load(self.mlp_path, map_location=self.device, weights_only=True))
            self.mnist_model.eval()

        # 2. Extract bounding box from canvas (Isolate the first drawing)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        
        # Dilate heavily to fuse nearby disconnected strokes (e.g., parts of a '5' or 'X') into a single blob
        kernel = np.ones((40, 40), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Find distinct drawings (blobs) on the canvas
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
            
        # Sort left-to-right so we use the "first one"
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        first_blob = contours[0]
        bx, by, bw, bh = cv2.boundingRect(first_blob)
        
        # Crop the original drawing using this blob's bounding box
        blob_crop = gray[by:by+bh, bx:bx+bw]
        
        # Find the TIGHT bounding box of the actual drawn pixels within this specific blob
        coords = cv2.findNonZero(blob_crop)
        if coords is None:
            return
            
        cx, cy, cw, ch = cv2.boundingRect(coords)
        x = bx + cx
        y = by + cy
        w = cw
        h = ch
        
        if w < 10 or h < 10:
            return
            
        # 3. Crop tight, pad to square, resize to 32x32
        digit_crop = gray[y:y+h, x:x+w]
        
        size = max(w, h)
        pad_x = (size - w) // 2
        pad_y = (size - h) // 2
        padded = np.pad(digit_crop, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        
        # Add dynamic padding (40% of size) to ensure the drawing NEVER touches the border
        pad_amt = int(size * 0.40)
        padded = np.pad(padded, ((pad_amt, pad_amt), (pad_amt, pad_amt)), mode='constant')
        
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Save 28x28 input for the OpenCV animation
        self.network_input = resized
        
        # 4. Predict
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tensor = transform(resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Step-by-Step Forward pass
            out_conv1 = self.mnist_model.relu1(self.mnist_model.conv1(tensor))
            out_pool1 = self.mnist_model.pool1(out_conv1)
            
            out_conv2 = self.mnist_model.relu2(self.mnist_model.conv2(out_pool1))
            out_pool2 = self.mnist_model.pool2(out_conv2)
            
            flat = self.mnist_model.flatten(out_pool2)
            out_fc1 = self.mnist_model.relu3(self.mnist_model.fc1(flat))
            out_fc2 = self.mnist_model.relu4(self.mnist_model.fc2(out_fc1))
            output = self.mnist_model.fc3(out_fc2)
            
            import torch.nn.functional as F
            self.hud_probs = F.softmax(output, dim=1)[0].cpu().numpy()
            
            pred = output.argmax(dim=1, keepdim=True).item()
        
        # --- Generate Activation Visualizations (Forward Pass Only) ---
        def normalize_act(act):
            a = act.cpu().numpy()[0]
            for i in range(a.shape[0]):
                if np.max(a[i]) > 0: a[i] = a[i] / np.max(a[i])
            return (a * 255).astype(np.uint8)

        # Conv1 Activations
        c1_act = normalize_act(out_conv1)
        c1_agrid = np.zeros((4*25, 8*25), dtype=np.uint8)
        for i in range(32):
            r, c = i // 8, i % 8
            c1_agrid[r*25:r*25+24, c*25:c*25+24] = c1_act[i]
        self.hud_c1_act = cv2.resize(cv2.applyColorMap(c1_agrid, cv2.COLORMAP_VIRIDIS), (288, 144), interpolation=cv2.INTER_NEAREST)
        
        # Conv2 Activations
        c2_act = normalize_act(out_conv2)
        c2_agrid = np.zeros((8*9, 8*9), dtype=np.uint8)
        for i in range(64):
            r, c = i // 8, i % 8
            c2_agrid[r*9:r*9+8, c*9:c*9+8] = c2_act[i]
        self.hud_c2_act = cv2.resize(cv2.applyColorMap(c2_agrid, cv2.COLORMAP_VIRIDIS), (176, 176), interpolation=cv2.INTER_NEAREST)

        self.prediction_result = pred
        self.anim_frame = 0