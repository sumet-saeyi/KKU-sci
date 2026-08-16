"""
Spaceship Dodge feature module.

Steer with hand X position, Closed_Fist to fire. Ported from the standalone
hand_spaceship_dodge.py prototype, but rewired onto the shared VisionWorker /
GestureBus instead of opening its own camera + MediaPipe pipeline -- this
feature never touches cv2.VideoCapture or mediapipe directly, it only reacts
to gesture_event / frame_ready signals like every other feature module.

Visual language ported from theme.md (Hand Spaceship Dodge Theme Spec):
cyan = open hand, coral = fist/fire/danger, HSL-random obstacles, bloom glow
on ship/bullets/obstacles, starfield + radial-gradient backdrop, rounded
obstacle corners.
"""
import colorsys
import json
import os
import random
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.feature_interface import FeatureModule
from core.gesture_bus import bus
from core.registry import register_feature

# ---------------- Theme (BGR tuples, since OpenCV expects BGR) ----------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_ACCENT = (255, 209, 79)     # cyan  #4fd1ff -- open hand / ship / values
COLOR_ACCENT2 = (127, 91, 255)    # coral #ff5b7f -- fist / fire / warning
COLOR_TEXT = (255, 242, 234)      # #eaf2ff
COLOR_MUTED = (200, 163, 143)     # #8fa3c8
COLOR_PANEL = (32, 18, 14)        # #0e1220
COLOR_FLAME = (79, 159, 255)      # #ff9f4f
COLOR_BULLET = (79, 225, 255)     # #ffe14f
COLOR_BG_CENTER = (48, 24, 16)    # #101830
COLOR_BG_EDGE = (10, 6, 5)        # #05060a
COLOR_STAGE_BORDER = (74, 48, 34) # #22304a
GLOW_SIGMA = 7

# ---------------- Game constants ----------------
# Sized to fill most of the content area on a 1536x960 screen (1536 - 200px
# sidebar = ~1336 usable) while still rendering 1:1, so the canvas is large
# and playable without being upscaled into blur.
GAME_W, GAME_H = 1120, 700
# Ship / speeds scaled with the canvas (~1.17x vs the old 960x600) so the
# playfield reads the same instead of everything looking shrunken.
SHIP_W, SHIP_H = 48, 48
SHIP_Y = GAME_H - 64
BULLET_SPEED = 14
BASE_OBSTACLE_SPEED = 3.0
FIRE_COOLDOWN_TICKS = 12
TICK_MS = 1000 // 60

# ---------------- Difficulty ramp ----------------
# Difficulty climbs with elapsed time (not just score), so simply surviving
# keeps raising the pressure: obstacles fall faster and spawn closer together.
TICKS_PER_SEC = 1000.0 / TICK_MS
DIFFICULTY_RAMP_SEC = 20.0   # one difficulty level per 20s survived
SPEED_PER_SEC = 0.055        # obstacle speed added per second survived
SPEED_PER_POINT = 0.015      # ...and per point scored
MAX_OBSTACLE_SPEED = 11.0    # ceiling, so it stays humanly playable
BASE_SPAWN_TICKS = 48        # gap between spawns at the start
SPAWN_DROP_PER_SEC = 0.55    # gap shrinks this much per second survived
SPAWN_DROP_PER_POINT = 0.12
MIN_SPAWN_TICKS = 14

# ---------------- Leaderboard ----------------
LEADERBOARD_SIZE = 10        # entries persisted to disk
LEADERBOARD_SHOWN = 5        # rows drawn on the game-over screen

CAM_PREVIEW_W, CAM_PREVIEW_H = 192, 144
CAM_PREVIEW_X, CAM_PREVIEW_Y = 16, 50


def random_obstacle_color_bgr():
    hue = random.uniform(300, 360) / 360
    r, g, b = colorsys.hls_to_rgb(hue, 0.60, 0.90)
    return (int(b * 255), int(g * 255), int(r * 255))


def rounded_rectangle(img, pt1, pt2, color, radius, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def apply_glow(frame, glow_layer, sigma=GLOW_SIGMA, downscale=2):
    # Blooms are low-frequency by nature, so blurring a half-res copy and
    # upscaling looks the same but costs ~4x less than blurring full-res.
    h, w = glow_layer.shape[:2]
    small = cv2.resize(glow_layer, (max(1, w // downscale), max(1, h // downscale)),
                        interpolation=cv2.INTER_LINEAR)
    blurred_small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / downscale)
    blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.add(frame, blurred)


def make_bg_gradient(w, h):
    """radial-gradient(circle at 50% 20%, #101830 0%, #05060a 70%)"""
    cx, cy = w * 0.5, h * 0.2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    radius = 0.7 * np.hypot(w, h)
    t = np.clip(dist / radius, 0, 1)[..., None]
    center = np.array(COLOR_BG_CENTER, dtype=np.float32)
    edge = np.array(COLOR_BG_EDGE, dtype=np.float32)
    grad = (1 - t) * center + t * edge
    return grad.astype(np.uint8)


class GameState:
    def __init__(self):
        self.best_score = 0
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = 3
        self.obstacles = []
        self.bullets = []
        self.spawn_timer = 0
        self.speed = BASE_OBSTACLE_SPEED
        self.game_over = False
        self.smoothed_ship_x = GAME_W / 2
        self.hand_x = GAME_W / 2
        self.was_fist = False
        self.fire_cooldown = 0
        self.ticks = 0
        self.level = 1

    @property
    def elapsed_sec(self):
        return self.ticks / TICKS_PER_SEC


def spawn_obstacle(state):
    w = random.randint(30, 80)
    h = random.randint(26, 44)
    x = random.randint(0, max(0, GAME_W - w))
    state.obstacles.append({"x": x, "y": -h, "w": w, "h": h, "color": random_obstacle_color_bgr()})


def fire_bullet(state):
    state.bullets.append({"x": state.smoothed_ship_x, "y": SHIP_Y - SHIP_H / 2})


def update_game(state, is_fist):
    state.ticks += 1
    elapsed = state.elapsed_sec
    state.level = 1 + int(elapsed / DIFFICULTY_RAMP_SEC)

    state.spawn_timer -= 1
    if state.spawn_timer <= 0:
        spawn_obstacle(state)
        state.spawn_timer = max(MIN_SPAWN_TICKS, int(
            BASE_SPAWN_TICKS
            - elapsed * SPAWN_DROP_PER_SEC
            - state.score * SPAWN_DROP_PER_POINT))

    state.speed = min(MAX_OBSTACLE_SPEED, BASE_OBSTACLE_SPEED
                      + elapsed * SPEED_PER_SEC
                      + state.score * SPEED_PER_POINT)

    state.smoothed_ship_x += (state.hand_x - state.smoothed_ship_x) * 0.22
    state.smoothed_ship_x = max(SHIP_W / 2, min(GAME_W - SHIP_W / 2, state.smoothed_ship_x))

    if state.fire_cooldown > 0:
        state.fire_cooldown -= 1
    if is_fist and not state.was_fist and state.fire_cooldown == 0:
        fire_bullet(state)
        state.fire_cooldown = FIRE_COOLDOWN_TICKS
    state.was_fist = is_fist

    state.bullets = [b for b in state.bullets if b["y"] > -20]
    for b in state.bullets:
        b["y"] -= BULLET_SPEED

    ship_left = state.smoothed_ship_x - SHIP_W / 2
    ship_right = state.smoothed_ship_x + SHIP_W / 2
    ship_top = SHIP_Y - SHIP_H / 2
    ship_bottom = SHIP_Y + SHIP_H / 2

    remaining = []
    for o in state.obstacles:
        o["y"] += state.speed

        hit_ship = (o["x"] < ship_right and o["x"] + o["w"] > ship_left
                    and o["y"] < ship_bottom and o["y"] + o["h"] > ship_top)
        if hit_ship:
            state.lives -= 1
            if state.lives <= 0:
                state.game_over = True
                state.best_score = max(state.best_score, state.score)
            continue

        destroyed = False
        for bidx, b in enumerate(state.bullets):
            if o["x"] < b["x"] < o["x"] + o["w"] and o["y"] < b["y"] < o["y"] + o["h"]:
                del state.bullets[bidx]
                state.score += 2
                destroyed = True
                break
        if destroyed:
            continue

        if o["y"] > GAME_H + 40:
            state.score += 1
            continue

        remaining.append(o)

    state.obstacles = remaining


@register_feature
class SpaceshipFeature(FeatureModule):
    def __init__(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.widget.setStyleSheet("background-color: #05060a;")

        self.display_label = QLabel("Spaceship Dodge")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ignored (not Preferred): the label must take its size from the layout,
        # never from its own pixmap. With Preferred, sizeHint follows the pixmap
        # while game_tick scales the pixmap to the label -- a feedback loop that
        # shrinks the view a little every frame until it collapses to a speck.
        # The render size is capped in game_tick instead (see NATIVE_SIZE).
        self.display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        main_layout.addWidget(self.display_label)

        self.widget.setLayout(main_layout)

        self.latest_cam_frame = None  # BGR numpy array, composited as a corner PiP inset

        self.state = GameState()
        _hub = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.scores_file = os.path.join(_hub, "spaceship_scores.json")
        self.legacy_best_file = os.path.join(_hub, "spaceship_best.json")
        self.leaderboard = self._load_scores()
        self.last_entry_rank = None  # row to highlight after a run
        self.state.best_score = self.leaderboard[0]["score"] if self.leaderboard else 0

        self.bg_gradient = make_bg_gradient(GAME_W, GAME_H)
        self.stars = [{
            "x": random.uniform(0, GAME_W),
            "y": random.uniform(0, GAME_H),
            "speed": random.uniform(0.2, 0.6),
            "v": random.randint(90, 200),
        } for _ in range(60)]

        self.hand_x_norm = 0.5
        self.is_fist = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.game_tick)
        self.timer.start(TICK_MS)

    @property
    def name(self) -> str:
        return "Spaceship Dodge"

    @property
    def icon(self) -> str:
        return "🚀"

    def build_widget(self) -> QWidget:
        bus.gesture_event.connect(self.on_gesture)
        bus.frame_ready.connect(self.update_camera_view)
        return self.widget

    def _load_scores(self):
        """Load the leaderboard, migrating the old single-best-score file so a
        previous personal best isn't thrown away."""
        if os.path.exists(self.scores_file):
            try:
                with open(self.scores_file, "r") as f:
                    entries = json.load(f).get("scores", [])
                clean = [e for e in entries if isinstance(e, dict) and "score" in e]
                clean.sort(key=lambda e: e.get("score", 0), reverse=True)
                return clean[:LEADERBOARD_SIZE]
            except Exception:
                return []

        if os.path.exists(self.legacy_best_file):
            try:
                with open(self.legacy_best_file, "r") as f:
                    best = int(json.load(f).get("best_score", 0))
                if best > 0:
                    return [{"score": best, "level": 1, "time": 0, "date": "-"}]
            except Exception:
                pass
        return []

    def _save_scores(self):
        try:
            with open(self.scores_file, "w") as f:
                json.dump({"scores": self.leaderboard}, f, indent=2)
        except Exception:
            pass

    def _record_score(self):
        """Insert the finished run into the leaderboard and persist it."""
        entry = {
            "score": self.state.score,
            "level": self.state.level,
            "time": int(self.state.elapsed_sec),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        self.leaderboard.append(entry)
        # Stable sort + appending last means an equal score ranks below the
        # existing holder, so ties don't bump an older run off its spot.
        self.leaderboard.sort(key=lambda e: e.get("score", 0), reverse=True)
        self.leaderboard = self.leaderboard[:LEADERBOARD_SIZE]

        # Identity match, not equality -- an identical older run must not steal
        # the highlight from the one just played.
        self.last_entry_rank = next(
            (i for i, e in enumerate(self.leaderboard) if e is entry), None)
        self.state.best_score = self.leaderboard[0]["score"] if self.leaderboard else 0
        self._save_scores()

    def update_camera_view(self, qt_img):
        if not self.widget.isVisible():
            return
        # Keep the raw camera frame as a numpy array so _render() can
        # composite it as a small corner inset on top of the game canvas,
        # instead of a separate side-panel widget eating screen space.
        qt_img = qt_img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qt_img.width(), qt_img.height()
        ptr = qt_img.bits()
        ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        self.latest_cam_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def on_gesture(self, event_data: dict):
        if not self.widget.isVisible():
            return

        frame_shape = event_data.get("frame_shape")
        frame_w = frame_shape[1] if frame_shape else 1280
        hands = event_data.get("hands", [])

        fist_now = False
        if hands:
            hand = hands[0]
            landmarks = hand.get("landmarks", [])
            if len(landmarks) > 9:
                self.hand_x_norm = landmarks[9][0] / frame_w
            fist_now = hand.get("gesture") == "Closed_Fist"

            if self.state.game_over and hand.get("gesture") == "Open_Palm":
                self.state.reset()
                self.last_entry_rank = None

        self.is_fist = fist_now

    def game_tick(self):
        if not self.widget.isVisible():
            return

        self.state.hand_x = self.hand_x_norm * GAME_W

        if not self.state.game_over:
            update_game(self.state, self.is_fist)
            if self.state.game_over:
                # Transition frame only -- update_game just flipped game_over to True.
                self._record_score()

        frame = self._render()
        h, w, ch = frame.shape
        bpl = ch * w
        qt_img = QImage(frame.data, w, h, bpl, QImage.Format.Format_RGB888)
        # Fill the available area, but never scale past the native render size
        # -- upscaling past 1:1 only adds blur. boundedTo() caps it; the label's
        # AlignCenter then centers the result.
        target = self.display_label.size().boundedTo(QSize(GAME_W, GAME_H))
        self.display_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _render(self):
        state = self.state
        canvas = self.bg_gradient.copy()

        for star in self.stars:
            star["y"] += star["speed"]
            if star["y"] > GAME_H:
                star["y"] = 0
                star["x"] = random.uniform(0, GAME_W)
            v = star["v"]
            cv2.circle(canvas, (int(star["x"]), int(star["y"])), 1, (v, v, v), -1, cv2.LINE_AA)

        glow = np.zeros_like(canvas)

        for o in state.obstacles:
            p1 = (int(o["x"]), int(o["y"]))
            p2 = (int(o["x"] + o["w"]), int(o["y"] + o["h"]))
            rounded_rectangle(canvas, p1, p2, o["color"], radius=6, thickness=-1)
            rounded_rectangle(canvas, p1, p2, (255, 255, 255), radius=6, thickness=1)
            rounded_rectangle(glow, p1, p2, o["color"], radius=6, thickness=-1)

        for b in state.bullets:
            pt = (int(b["x"]), int(b["y"]))
            cv2.circle(canvas, pt, 4, COLOR_BULLET, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, COLOR_BULLET, 1, cv2.LINE_AA)
            cv2.circle(glow, pt, 4, COLOR_BULLET, -1, cv2.LINE_AA)

        x, y = int(state.smoothed_ship_x), int(SHIP_Y)
        pts = np.array([[x, y - 26], [x + 19, y + 22], [x, y + 12], [x - 19, y + 22]], np.int32)
        flame_h = 31 + random.randint(0, 7)
        flame = np.array([[x - 6, y + 19], [x, y + flame_h], [x + 6, y + 19]], np.int32)
        for target in (canvas, glow):
            cv2.fillPoly(target, [pts], COLOR_ACCENT, cv2.LINE_AA)
            cv2.fillPoly(target, [flame], COLOR_FLAME, cv2.LINE_AA)

        canvas = apply_glow(canvas, glow)

        marker_color = COLOR_ACCENT2 if self.is_fist else COLOR_ACCENT
        overlay = canvas.copy()
        cv2.line(overlay, (int(state.hand_x), 46), (int(state.hand_x), SHIP_Y), marker_color, 2, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

        self._draw_hud(canvas)
        self._draw_camera_preview(canvas)
        if state.game_over:
            self._draw_game_over(canvas)

        rounded_rectangle(canvas, (0, 0), (GAME_W - 1, GAME_H - 1), COLOR_STAGE_BORDER, radius=16, thickness=2)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _draw_camera_preview(self, canvas):
        """Small camera inset in the corner (per theme.md: 160x120, rounded,
        border color follows gesture) instead of a separate side-panel widget."""
        if self.latest_cam_frame is None:
            return
        pw, ph = CAM_PREVIEW_W, CAM_PREVIEW_H
        x0, y0 = CAM_PREVIEW_X, CAM_PREVIEW_Y
        thumb = cv2.resize(self.latest_cam_frame, (pw, ph))

        mask = np.zeros((ph, pw), dtype=np.uint8)
        rounded_rectangle(mask, (0, 0), (pw - 1, ph - 1), 255, radius=10, thickness=-1)
        inv_mask = cv2.bitwise_not(mask)

        roi = canvas[y0:y0 + ph, x0:x0 + pw]
        thumb_masked = cv2.bitwise_and(thumb, thumb, mask=mask)
        roi_masked = cv2.bitwise_and(roi, roi, mask=inv_mask)
        canvas[y0:y0 + ph, x0:x0 + pw] = cv2.add(roi_masked, thumb_masked)

        border_color = COLOR_ACCENT2 if self.is_fist else COLOR_ACCENT
        rounded_rectangle(canvas, (x0, y0), (x0 + pw - 1, y0 + ph - 1), border_color, radius=10, thickness=2)

    def _draw_stat(self, canvas, x, y, label, value, value_color):
        cv2.putText(canvas, label, (x, y), FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        label_w = cv2.getTextSize(label, FONT, 0.5, 1)[0][0]
        cv2.putText(canvas, str(value), (x + label_w + 6, y), FONT, 0.55, value_color, 2, cv2.LINE_AA)

    def _draw_hud(self, canvas):
        state = self.state
        bar_h = 40
        cv2.rectangle(canvas, (0, 0), (GAME_W, bar_h), COLOR_PANEL, -1)
        self._draw_stat(canvas, 14, 26, "Score: ", state.score, COLOR_ACCENT)
        self._draw_stat(canvas, 160, 26, "Best: ", state.best_score, COLOR_ACCENT)
        self._draw_stat(canvas, 290, 26, "Lives: ", state.lives, COLOR_ACCENT2)

        elapsed = int(state.elapsed_sec)
        self._draw_stat(canvas, 420, 26, "Level: ", state.level, COLOR_FLAME)
        self._draw_stat(canvas, 545, 26, "Time: ",
                        f"{elapsed // 60}:{elapsed % 60:02d}", COLOR_TEXT)

        # Progress toward the next difficulty level, as a thin bar under the HUD.
        frac = (state.elapsed_sec % DIFFICULTY_RAMP_SEC) / DIFFICULTY_RAMP_SEC
        cv2.rectangle(canvas, (0, bar_h), (GAME_W, bar_h + 3), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (0, bar_h), (int(GAME_W * frac), bar_h + 3), COLOR_FLAME, -1)

        gesture_txt = "FIST - FIRE!" if self.is_fist else "OPEN HAND"
        gesture_color = COLOR_ACCENT2 if self.is_fist else COLOR_ACCENT
        text_w = cv2.getTextSize(gesture_txt, FONT, 0.5, 2)[0][0]
        cv2.putText(canvas, gesture_txt, (GAME_W - text_w - 14, 26), FONT, 0.5, gesture_color, 2, cv2.LINE_AA)

    def _centered_text(self, canvas, text, cx, y, scale, color, thickness):
        tw = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
        cv2.putText(canvas, text, (cx - tw // 2, y), FONT, scale, color, thickness, cv2.LINE_AA)

    def _draw_game_over(self, canvas):
        state = self.state
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (GAME_W, GAME_H), (0, 0, 0), -1)
        canvas[:] = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

        cx, cy = GAME_W // 2, GAME_H // 2
        elapsed = int(state.elapsed_sec)

        self._centered_text(canvas, "GAME OVER", cx, cy - 150, 1.2, COLOR_ACCENT2, 3)
        summary = (f"Score: {state.score}   Level: {state.level}   "
                   f"Time: {elapsed // 60}:{elapsed % 60:02d}")
        self._centered_text(canvas, summary, cx, cy - 112, 0.6, COLOR_TEXT, 2)

        self._draw_leaderboard(canvas, cx, cy - 66)

        self._centered_text(canvas, "Show an open palm to restart", cx,
                            cy + 178, 0.5, COLOR_MUTED, 1)

    def _draw_leaderboard(self, canvas, cx, top_y):
        """Top scores table, with the run just played highlighted."""
        rows = self.leaderboard[:LEADERBOARD_SHOWN]
        table_w = 560
        x0 = cx - table_w // 2

        # Column x-offsets within the table (rank, score, level, time, date)
        col_rank = x0 + 14
        col_score = x0 + 70
        col_level = x0 + 190
        col_time = x0 + 290
        col_date = x0 + 390

        self._centered_text(canvas, "LEADERBOARD", cx, top_y, 0.55, COLOR_ACCENT, 2)

        header_y = top_y + 28
        for label, lx in (("#", col_rank), ("SCORE", col_score), ("LEVEL", col_level),
                          ("TIME", col_time), ("DATE", col_date)):
            cv2.putText(canvas, label, (lx, header_y), FONT, 0.4, COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, header_y + 8), (x0 + table_w, header_y + 8),
                 COLOR_STAGE_BORDER, 1, cv2.LINE_AA)

        if not rows:
            self._centered_text(canvas, "No runs yet -- set the first score!", cx,
                                header_y + 40, 0.45, COLOR_MUTED, 1)
            return

        row_h = 26
        for i, entry in enumerate(rows):
            ry = header_y + 34 + i * row_h
            is_new = (i == self.last_entry_rank)
            color = COLOR_FLAME if is_new else COLOR_TEXT

            if is_new:
                cv2.rectangle(canvas, (x0, ry - 15), (x0 + table_w, ry + 7),
                              COLOR_PANEL, -1)

            secs = int(entry.get("time", 0))
            # Drop the year so the date column doesn't run into the NEW badge.
            date_txt = str(entry.get("date", "-"))
            if len(date_txt) >= 16:
                date_txt = date_txt[5:]
            cells = (
                (f"{i + 1}", col_rank),
                (str(entry.get("score", 0)), col_score),
                (str(entry.get("level", 1)), col_level),
                (f"{secs // 60}:{secs % 60:02d}", col_time),
                (date_txt, col_date),
            )
            for text, lx in cells:
                cv2.putText(canvas, text, (lx, ry), FONT, 0.45, color, 1, cv2.LINE_AA)

            if is_new:
                cv2.putText(canvas, "NEW", (x0 + table_w - 44, ry), FONT, 0.4,
                            COLOR_ACCENT2, 1, cv2.LINE_AA)
