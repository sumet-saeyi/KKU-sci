import os
import random
import time
import json
import pygame
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QInputDialog, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QPainter, QFont, QColor
from PyQt6.QtCore import Qt, QTimer
from core.feature_interface import FeatureModule
from core.registry import register_feature
from core.gesture_bus import bus

# VARIABLES (Tuned for 60 FPS)
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 5.0
GRAVITY = 0.15625
GAME_SPEED = 3.75

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT= 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500

PIPE_GAP = 250

current_dir = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(current_dir, "flappy_assets")

pygame.init()
pygame.display.set_mode((1, 1), pygame.HIDDEN)
pygame.mixer.init()

wing = os.path.join(ASSETS, 'audio', 'wing.wav')
hit = os.path.join(ASSETS, 'audio', 'hit.wav')

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images =  [pygame.image.load(os.path.join(ASSETS, 'sprites', 'KKU-upflap.png')).convert_alpha(),
                        pygame.image.load(os.path.join(ASSETS, 'sprites', 'KKU-midflap.png')).convert_alpha(),
                        pygame.image.load(os.path.join(ASSETS, 'sprites', 'KKU-downflap.png')).convert_alpha()]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.y_pos = float(SCREEN_HEIGHT / 2)
        self.rect[1] = int(self.y_pos)

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.y_pos += self.speed
        self.rect[1] = int(self.y_pos)

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(ASSETS, 'sprites', 'pipe-green.png')).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.x_pos = float(xpos)
        self.rect[0] = int(self.x_pos)
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.x_pos -= GAME_SPEED
        self.rect[0] = int(self.x_pos)

class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(ASSETS, 'sprites', 'base.png')).convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.x_pos = float(xpos)
        self.rect[0] = int(self.x_pos)
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.x_pos -= GAME_SPEED
        self.rect[0] = int(self.x_pos)

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

@register_feature
class FlappyBirdFeature(FeatureModule):
    def __init__(self):
        self.widget = QWidget()
        self.main_layout = QHBoxLayout()
        
        self.widget.setStyleSheet("background-color: #0f172a;")
        
        self.display_label = QLabel("Flappy Bird")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.display_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.8);
                border: 2px solid #0ea5e9;
                border-radius: 12px;
            }
        """)
        self.main_layout.addWidget(self.display_label, 1)
        
        self.right_layout = QVBoxLayout()
        
        self.cam_label = QLabel("Camera")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.cam_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.5);
                border: 1px solid #0284c7;
                border-radius: 12px;
            }
        """)
        self.right_layout.addWidget(self.cam_label, 1)
        
        self.scoreboard_label = QLabel("Scoreboard")
        self.scoreboard_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.scoreboard_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.9);
                border: 1px solid #f59e0b;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        self.right_layout.addWidget(self.scoreboard_label, 1)
        
        self.main_layout.addLayout(self.right_layout, 1)
        self.widget.setLayout(self.main_layout)
        
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        self.background_img = pygame.image.load(os.path.join(ASSETS, 'sprites', 'background-day.png'))
        self.background_img = pygame.transform.scale(self.background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Cyber tint for the background
        self.tint = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.tint.fill((14, 165, 233)) # Teal
        self.tint.set_alpha(60)
        
        self.begin_image = pygame.image.load(os.path.join(ASSETS, 'sprites', 'message.png')).convert_alpha()
        
        try:
            self.gameover_image = pygame.image.load(os.path.join(ASSETS, 'sprites', 'gameover.png')).convert_alpha()
        except:
            pass # Fallback handled if needed
        
        self.top_left_qr = QImage(os.path.join(ASSETS, 'sprites', 'top_left_qr.png'))
        self.top_right_qr = QImage(os.path.join(ASSETS, 'sprites', 'top_right_qr.png'))
        
        self.scores_file = os.path.join(current_dir, "flappy_scores.json")
        self.top_scores = self.load_scores()
        
        self.reset_game()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_tick)
        self.timer.start(1000 // 60) # 60 FPS
        
        self.last_jump_time = 0
        self.hand_was_open = False
        self.scroll_offset = 0

    @property
    def name(self):
        return "Flappy Bird"

    @property
    def icon(self):
        return "🐦"
        
    def load_scores(self):
        if os.path.exists(self.scores_file):
            try:
                with open(self.scores_file, "r") as f:
                    scores = json.load(f)
                    return [{"name": "Anonymous", "score": s} if isinstance(s, int) else s for s in scores]
            except:
                return []
        return []

    def save_score(self):
        if self.score > 0:
            name = time.strftime("%b %d, %H:%M")
            self.top_scores.append({"name": name, "score": self.score})
            
            self.top_scores.sort(key=lambda x: x["score"], reverse=True)
            self.top_scores = self.top_scores[:10] # Top 10 only
            try:
                with open(self.scores_file, "w") as f:
                    json.dump(self.top_scores, f)
            except:
                pass
        
    def reset_game(self):
        self.state = "start"
        self.score = 0
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)
        
        self.ground_group = pygame.sprite.Group()
        for i in range(2):
            ground = Ground(GROUND_WIDTH * i)
            self.ground_group.add(ground)
            
        self.pipe_group = pygame.sprite.Group()
        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

    def build_widget(self):
        bus.gesture_event.connect(self.on_gesture)
        bus.frame_ready.connect(self.update_camera_view)
        return self.widget

    def update_camera_view(self, qt_img):
        if not self.widget.isVisible():
            return
            
        painter = QPainter(qt_img)
            
        # Draw AR Cyber HUD overlay on camera
        # painter.setBrush(QColor(15, 23, 42, 180)) # Semi-transparent dark background
        # painter.setPen(Qt.PenStyle.NoPen)
        # painter.drawRoundedRect(15, 15, 420, 165, 15, 15)
        
        # painter.setPen(QColor(245, 158, 11)) # Amber
        # painter.setFont(QFont("Consolas", 21, QFont.Weight.Bold))
        # painter.drawText(30, 60, ":: COMMAND ::")

        # don't need it      
        # painter.setPen(QColor(14, 165, 233)) # Teal
        # painter.setFont(QFont("Consolas", 18))
        # painter.drawText(30, 112, "🖐️ Open Palm = Jump")
        # painter.drawText(30, 157, "⏱️ Auto-reset on death")
        
        img_w = qt_img.width()
        
        # --- Top Right QR ---
        # if not self.top_right_qr.isNull():
        #     # Create a 200x200 box for the QR
        #     scaled_tr = self.top_right_qr.scaled(450, 450, Qt.AspectRatioMode.KeepAspectRatio)
        #     tr_x = img_w - scaled_tr.width() - 20
        #     tr_y = 20
        #     painter.drawImage(tr_x, tr_y, scaled_tr)
        #     # Dummy text under QR
        #     painter.setPen(QColor(255, 255, 255))
        #     painter.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        #     painter.drawText(tr_x, tr_y + scaled_tr.height() + 25, "DUMMY TEXT RIGHT")
            
        # --- Top Left QR ---
        # if not self.top_left_qr.isNull():
        #     # Create a 200x200 box for the QR
        #     scaled_tl = self.top_left_qr.scaled(450, 450, Qt.AspectRatioMode.KeepAspectRatio)
        #     tl_x = 20
        #     tl_y = 20 # Placed below the HUD box (which ends at Y=180)
        #     painter.drawImage(tl_x, tl_y, scaled_tl)
        #     # Dummy text under QR
        #     painter.setPen(QColor(255, 255, 255))
        #     painter.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        #     painter.drawText(tl_x, tl_y + scaled_tl.height() + 25, "DUMMY TEXT LEFT")
            
        painter.end()

        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.cam_label.setPixmap(scaled_pixmap)

    def on_gesture(self, event_data: dict):
        if not self.widget.isVisible():
            return
            
        hands = event_data.get('hands', [])
        now = time.time()
        
        jumped = False
        any_clear = False
        
        for hand in hands:
            command = hand.get('command', 'idle')
            h_id = hand.get('handedness', 'Unknown')
            
            # Use Open hand (clear) to jump! (Trigger only once per open motion)
            if command == "stop":
                any_clear = True
                if not self.hand_was_open:
                    jumped = True
                
        self.hand_was_open = any_clear
                
        if jumped and self.state != "gameover":
            self.bird.bump()
            try:
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()
            except:
                pass
            if self.state == "start":
                self.state = "playing"
            self.last_jump_time = now

    def game_tick(self):
        if not self.widget.isVisible():
            return
            
        if self.state == "gameover":
            if time.time() - getattr(self, 'gameover_time', time.time()) > 1.0:
                self.reset_game()
                return
        elif self.state == "start":
            self.bird.begin()
            self.ground_group.update()
            if is_off_screen(self.ground_group.sprites()[0]):
                self.ground_group.remove(self.ground_group.sprites()[0])
                self.ground_group.add(Ground(GROUND_WIDTH - 20))
        elif self.state == "playing":
            if is_off_screen(self.ground_group.sprites()[0]):
                self.ground_group.remove(self.ground_group.sprites()[0])
                self.ground_group.add(Ground(GROUND_WIDTH - 20))
                
            if is_off_screen(self.pipe_group.sprites()[0]):
                self.pipe_group.remove(self.pipe_group.sprites()[0])
                self.pipe_group.remove(self.pipe_group.sprites()[0])
                pipes = get_random_pipes(SCREEN_WIDTH * 2)
                self.pipe_group.add(pipes[0])
                self.pipe_group.add(pipes[1])
                self.score += 1
                
            self.bird_group.update()
            self.ground_group.update()
            self.pipe_group.update()
            
            if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)):
                try:
                    pygame.mixer.music.load(hit)
                    pygame.mixer.music.play()
                except:
                    pass
                self.state = "gameover"
                self.gameover_time = time.time()
                self.save_score()

        # Render
        self.surface.blit(self.background_img, (0, 0))
        self.surface.blit(self.tint, (0, 0))
        
        if self.state == "start":
            self.surface.blit(self.begin_image, (120, 150))
            
        self.pipe_group.draw(self.surface)
        self.ground_group.draw(self.surface)
        self.bird_group.draw(self.surface)
        
        if self.state == "gameover":
            if hasattr(self, 'gameover_image'):
                go_w = self.gameover_image.get_width()
                go_h = self.gameover_image.get_height()
                self.surface.blit(self.gameover_image, (SCREEN_WIDTH // 2 - go_w // 2, SCREEN_HEIGHT // 2 - go_h // 2))
            else:
                pygame.font.init()
                go_font = pygame.font.SysFont('Consolas', 60, True)
                go_text = go_font.render("GAME OVER", True, (239, 68, 68))
                go_outline = go_font.render("GAME OVER", True, (0, 0, 0))
                self.surface.blit(go_outline, (SCREEN_WIDTH // 2 - go_text.get_width() // 2 + 3, SCREEN_HEIGHT // 2 - go_text.get_height() // 2 + 3))
                self.surface.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, SCREEN_HEIGHT // 2 - go_text.get_height() // 2))
        
        # Draw Score
        if self.state in ["playing", "gameover"]:
            pygame.font.init()
            font = pygame.font.SysFont('Consolas', 48, True)
            text = font.render(str(self.score), True, (255, 255, 255))
            # Outline
            outline = font.render(str(self.score), True, (0, 0, 0))
            self.surface.blit(outline, (SCREEN_WIDTH // 2 - text.get_width() // 2 + 2, 50 + 2))
            self.surface.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 50))

        arr = pygame.surfarray.array3d(self.surface)
        arr = np.transpose(arr, (1, 0, 2))
        h, w, ch = arr.shape
        bpl = ch * w
        arr_contiguous = np.ascontiguousarray(arr)
        qt_img = QImage(arr_contiguous.data, w, h, bpl, QImage.Format.Format_RGB888)
        
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.display_label.setPixmap(scaled_pixmap)
        
        # Handle auto-scrolling scoreboard with HTML Cyber styling
        self.scroll_offset += 1
        
        lines = [
            "<div style='font-family: Consolas;'>",
            "<center><span style='color: #f59e0b; font-size: 24px; font-weight: bold; margin-bottom: 10px;'>🏆 LEADERBOARD 🏆</span></center>",
            "<hr style='border: 1px solid #0284c7; width: 80%; margin: 10px auto;'>",
            "<table width='100%' style='font-size: 18px; margin-top: 10px;'>"
        ]
        
        if not self.top_scores:
            lines.append("<tr><td colspan='2' style='color: #94a3b8; text-align: center;'>No scores yet!</td></tr>")
        else:
            for i, s in enumerate(self.top_scores):
                # Top score is glowing TEAL, others are TEAL DIM
                color = "#0ea5e9" if i == 0 else "#0284c7"
                name_color = "#e2e8e8"
                score_color = "#f59e0b" # AMBER
                lines.append(f"<tr><td style='color: {color}; width: 30px;'><b>{i+1}.</b></td><td style='color: {name_color};'>{s['name']}</td><td style='color: {score_color}; text-align: right; font-weight: bold;'>{s['score']}</td></tr>")
                
        lines.append("</table></div>")
        
        # We don't even need to slice lines because HTML handles layout cleanly!
        # But we'll still use the scroll_offset for a cool blinking effect on the header
        if self.scroll_offset % 30 < 15:
            lines[1] = "<center><span style='color: #f59e0b; font-size: 24px; font-weight: bold; margin-bottom: 10px;'>🏆 KKU FLAPER LEADERBOARD 🏆</span></center>"
        else:
            lines[1] = "<center><span style='color: #d97706; font-size: 24px; font-weight: bold; margin-bottom: 10px;'>🏆 KKU FLAPER LEADERBOARD 🏆</span></center>"

        self.scoreboard_label.setText("".join(lines))
