"""
Thai Digit Dataset Collector
====================================================
Draw Thai numerals ๐-๙ on a 400×400 canvas.
Saves 32×32 black/white PNGs organised by digit folder.

Output structure:
  data/thai_digits/
    0/  →  0-1.png, 0-2.png, …
    …
    9/  →  9-1.png, 9-2.png, …

Requirements: pip install Pillow
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk

# ── Config ────────────────────────────────────────────────────────────────────
CANVAS_SIZE = 400          # Drawing zone size (pixels)
OUTPUT_SIZE = 32           # Saved image size (32×32)
BRUSH_RADIUS = 10          # Brush radius on the 400×400 canvas
THAI_DIGITS = ["๐", "๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙"]
TARGET_PER_DIGIT = 5

# Resolve output directory relative to the script / exe
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "thai_digits")

# ── Colours (Catppuccin Mocha) ────────────────────────────────────────────────
BG        = "#1e1e2e"
BG_LIGHT  = "#313244"
SURFACE   = "#45475a"
TEXT      = "#cdd6f4"
BLUE      = "#89b4fa"
GREEN     = "#a6e3a1"
YELLOW    = "#f9e2af"
RED       = "#f38ba8"
MAUVE     = "#cba6f7"


class ThaiDigitCollector:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thai Digit Dataset Collector")
        self.root.configure(bg=BG)

        self.selected_digit = tk.IntVar(value=0)
        self._is_fullscreen = False
        self.strokes: list[list[tuple[int, int]]] = []
        self._current_stroke: list[tuple[int, int]] = []
        self._last_point: tuple[int, int] | None = None
        self.brush_size = tk.IntVar(value=BRUSH_RADIUS)

        # Track current canvas size (starts at CANVAS_SIZE, grows in fullscreen)
        self.canvas_size = CANVAS_SIZE
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._refresh_progress()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.main_frame = tk.Frame(self.root, bg=BG)
        self.main_frame.pack(fill="both", expand=True)

        # ── LEFT: Drawing canvas (400×400 default, expands in fullscreen) ────
        self.canvas_outer = tk.Frame(self.main_frame, bg=BG)
        self.canvas_outer.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        self.canvas_title = tk.Label(
            self.canvas_outer, text=f"Drawing Zone  ({CANVAS_SIZE}×{CANVAS_SIZE}  →  {OUTPUT_SIZE}×{OUTPUT_SIZE})",
            bg=BG, fg=TEXT, font=("Segoe UI", 10))
        self.canvas_title.pack(pady=(0, 4))

        # Center frame to hold the square canvas
        self.canvas_center = tk.Frame(self.canvas_outer, bg=BG)
        self.canvas_center.pack(fill="both", expand=True)

        canvas_border = tk.Frame(self.canvas_center, bg=SURFACE, bd=0)
        canvas_border.place(relx=0.5, rely=0.5, anchor="center",
                            width=CANVAS_SIZE + 4, height=CANVAS_SIZE + 4)
        self._canvas_border = canvas_border

        self.canvas = tk.Canvas(
            canvas_border, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=2, highlightbackground=SURFACE,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<B1-Motion>", self._on_draw)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Watch the outer frame for resize, keep canvas square
        self.canvas_center.bind("<Configure>", self._on_outer_resize)

        # ── RIGHT: Sidebar ────────────────────────────────────────────────────
        sidebar_width = 300
        sidebar = tk.Frame(self.main_frame, bg=BG, width=sidebar_width)
        sidebar.pack(side="right", fill="y", padx=(8, 16), pady=16)
        sidebar.pack_propagate(False)

        # Title
        tk.Label(sidebar, text="✍️ Thai Digit\nDataset Collector",
                 bg=BG, fg=BLUE, font=("Segoe UI", 16, "bold"),
                 justify="center").pack(pady=(10, 5))

        tk.Frame(sidebar, bg=SURFACE, height=2).pack(fill="x", padx=10, pady=5)

        # ── Digit selector (2 rows of 5) ─────────────────────────────────────
        tk.Label(sidebar, text="เลือกตัวเลข (0-9):", bg=BG, fg=TEXT,
                 font=("Segoe UI", 11)).pack(anchor="w", padx=14, pady=(8, 4))

        self.digit_buttons: list[tk.Button] = []
        for row_start in (0, 5):
            row_frame = tk.Frame(sidebar, bg=BG)
            row_frame.pack(padx=10, pady=2)
            for d in range(row_start, row_start + 5):
                b = tk.Button(
                    row_frame,
                    text=f"{THAI_DIGITS[d]}\n{d}",
                    width=4, height=2,
                    font=("Segoe UI", 12, "bold"),
                    bg=BG_LIGHT, fg=TEXT,
                    activebackground=MAUVE, activeforeground=BG,
                    relief="flat", bd=0, cursor="hand2",
                    command=lambda digit=d: self._select_digit(digit),
                )
                b.pack(side="left", padx=3, pady=2)
                self.digit_buttons.append(b)

        self._highlight_selected_button()

        tk.Frame(sidebar, bg=SURFACE, height=2).pack(fill="x", padx=10, pady=8)

        # ── 32×32 Preview ────────────────────────────────────────────────────
        tk.Label(sidebar, text=f"{OUTPUT_SIZE}×{OUTPUT_SIZE} Preview", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10)).pack(pady=(0, 2))

        preview_container = tk.Frame(sidebar, bg=SURFACE, bd=2, relief="groove")
        preview_container.pack(pady=(0, 4))
        self.preview_label = tk.Label(preview_container, bg="black",
                                       width=128, height=128)
        self.preview_label.pack(padx=2, pady=2)

        # ── Current digit info ───────────────────────────────────────────────
        self.digit_info_var = tk.StringVar(value="")
        tk.Label(sidebar, textvariable=self.digit_info_var, bg=BG, fg=YELLOW,
                 font=("Segoe UI", 12, "bold"), justify="center").pack(pady=4)

        tk.Frame(sidebar, bg=SURFACE, height=2).pack(fill="x", padx=10, pady=4)

        # ── Action buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(sidebar, bg=BG)
        btn_frame.pack(pady=6, padx=10, fill="x")

        for text, cmd, color in [
            ("💾  SAVE  (Ctrl+S)", self._save, GREEN),
            ("↩️  Undo  (Ctrl+Z)", self._undo, YELLOW),
            ("🗑️  Clear  (Del)", self._clear, RED),
        ]:
            tk.Button(
                btn_frame, text=text, command=cmd,
                bg=color, fg=BG, font=("Segoe UI", 11, "bold"),
                activebackground=color, relief="flat",
                padx=10, pady=6, cursor="hand2",
            ).pack(fill="x", pady=3)

        tk.Frame(sidebar, bg=SURFACE, height=2).pack(fill="x", padx=10, pady=4)

        # ── Brush size slider ─────────────────────────────────────────────────
        self.brush_label_var = tk.StringVar(value=f"🖌️  Brush Size: {BRUSH_RADIUS}")
        tk.Label(sidebar, textvariable=self.brush_label_var, bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(pady=(0, 2))

        brush_frame = tk.Frame(sidebar, bg=BG)
        brush_frame.pack(padx=10, fill="x", pady=(0, 2))

        tk.Button(brush_frame, text="  −  ", font=("Segoe UI", 10, "bold"),
                  bg=BG_LIGHT, fg=TEXT, relief="flat", cursor="hand2",
                  command=lambda: self._adjust_brush(-2)).pack(side="left", padx=(0, 4))

        self.brush_slider = tk.Scale(
            brush_frame, from_=2, to=30, orient="horizontal",
            variable=self.brush_size, showvalue=False,
            bg=BG, fg=TEXT, troughcolor=BG_LIGHT, activebackground=MAUVE,
            highlightthickness=0, bd=0, sliderlength=20,
            command=lambda v: self.brush_label_var.set(f"🖌️  Brush Size: {v}"),
        )
        self.brush_slider.pack(side="left", fill="x", expand=True, padx=4)

        tk.Button(brush_frame, text="  +  ", font=("Segoe UI", 10, "bold"),
                  bg=BG_LIGHT, fg=TEXT, relief="flat", cursor="hand2",
                  command=lambda: self._adjust_brush(2)).pack(side="left", padx=(4, 0))

        tk.Frame(sidebar, bg=SURFACE, height=2).pack(fill="x", padx=10, pady=6)

        # ── Progress table ───────────────────────────────────────────────────
        tk.Label(sidebar, text="📊  Dataset Progress", bg=BG, fg=BLUE,
                 font=("Segoe UI", 12, "bold")).pack(pady=(0, 4))

        progress_frame = tk.Frame(sidebar, bg=BG)
        progress_frame.pack(padx=10, fill="x")

        self.progress_labels: list[tk.Label] = []
        for d in range(10):
            row = tk.Frame(progress_frame, bg=BG)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=f"{THAI_DIGITS[d]} ({d})", bg=BG, fg=TEXT,
                     font=("Segoe UI", 9), width=6, anchor="w").pack(side="left")

            lbl = tk.Label(row, text="", bg=BG, fg=GREEN,
                           font=("Consolas", 9), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            self.progress_labels.append(lbl)

        self.total_label = tk.Label(sidebar, text="", bg=BG, fg=BLUE,
                                     font=("Segoe UI", 11, "bold"))
        self.total_label.pack(pady=(6, 4))

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Draw a Thai digit and press Save!  |  F11 = Fullscreen")
        tk.Label(sidebar, textvariable=self.status_var, bg=BG, fg=SURFACE,
                 font=("Segoe UI", 9), wraplength=260).pack(side="bottom", pady=(0, 6))

        # ── Keyboard shortcuts ───────────────────────────────────────────────
        self.root.bind("<Control-s>", lambda e: self._save())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Delete>", lambda e: self._clear())
        self.root.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.root.bind("<bracketleft>", lambda e: self._adjust_brush(-2))
        self.root.bind("<bracketright>", lambda e: self._adjust_brush(2))
        for d in range(10):
            self.root.bind(str(d), lambda e, digit=d: self._select_digit(digit))

    # ── Fullscreen ────────────────────────────────────────────────────────────
    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.root.attributes("-fullscreen", self._is_fullscreen)

    def _exit_fullscreen(self):
        if self._is_fullscreen:
            self._is_fullscreen = False
            self.root.attributes("-fullscreen", False)
        else:
            self.root.destroy()

    # ── Canvas resize (keep square) ─────────────────────────────────────────
    def _on_outer_resize(self, event):
        """Keep canvas square: use min(width, height) of the outer frame."""
        new_size = min(event.width, event.height)
        if new_size > 50 and abs(new_size - self.canvas_size) > 5:
            self._resize_canvas(new_size)

    def _resize_canvas(self, new_size: int):
        old_size = self.canvas_size
        self.canvas_size = new_size

        # Resize the border frame (place geometry) to stay square
        self._canvas_border.place_configure(
            width=new_size + 4, height=new_size + 4)

        # Scale existing strokes to new size
        if old_size > 0 and self.strokes:
            scale = new_size / old_size
            self.strokes = [
                [(int(x * scale), int(y * scale)) for x, y in stroke]
                for stroke in self.strokes
            ]

        # Reinitialise PIL image and redraw
        self.pil_image = Image.new("L", (new_size, new_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self._redraw_from_strokes()
        self._update_preview()
        self.canvas_title.configure(
            text=f"Drawing Zone  ({new_size}×{new_size}  →  {OUTPUT_SIZE}×{OUTPUT_SIZE})")

    # ── Drawing callbacks ─────────────────────────────────────────────────────
    def _on_press(self, event):
        self._current_stroke = []
        self._last_point = (event.x, event.y)
        self._draw_point(event.x, event.y)

    def _on_draw(self, event):
        if self._last_point:
            self._draw_line(self._last_point[0], self._last_point[1], event.x, event.y)
        else:
            self._draw_point(event.x, event.y)
        self._last_point = (event.x, event.y)

    def _on_release(self, _event):
        if self._current_stroke:
            self.strokes.append(self._current_stroke)
            self._current_stroke = []
        self._last_point = None
        self._update_preview()

    def _brush_r(self) -> int:
        """User-adjustable brush radius, scaled with canvas size."""
        return max(2, int(self.brush_size.get() * self.canvas_size / CANVAS_SIZE))

    def _adjust_brush(self, delta: int):
        new_val = max(2, min(30, self.brush_size.get() + delta))
        self.brush_size.set(new_val)
        self.brush_label_var.set(f"🖌️  Brush Size: {new_val}")

    def _draw_point(self, x: int, y: int):
        r = self._brush_r()
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self._current_stroke.append((x, y))

    def _draw_line(self, x0: int, y0: int, x1: int, y1: int):
        r = self._brush_r()
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=r * 2,
                                capstyle="round", joinstyle="round")
        self.pil_draw.line([(x0, y0), (x1, y1)], fill=255, width=r * 2)
        self.pil_draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)
        self._current_stroke.append((x1, y1))

    # ── Actions ───────────────────────────────────────────────────────────────
    def _save(self):
        if self.pil_image.getbbox() is None:
            messagebox.showwarning("Empty Canvas", "กรุณาวาดตัวเลขก่อนบันทึก")
            return

        digit = self.selected_digit.get()
        folder = os.path.join(OUTPUT_DIR, str(digit))
        os.makedirs(folder, exist_ok=True)

        existing = self._count_images(digit)
        seq = existing + 1
        filename = f"{digit}-{seq}.png"
        filepath = os.path.join(folder, filename)

        resized = self._crop_and_resize(self.pil_image)
        resized.save(filepath)

        self._clear()
        self._refresh_progress()

        count = self._count_images(digit)
        self.status_var.set(f"✅ Saved {filename}  ({count}/{TARGET_PER_DIGIT})")
        self.root.after(3000, lambda: self.status_var.set(
            "Draw a Thai digit and press Save!  |  F11 = Fullscreen"))

    def _undo(self):
        if not self.strokes:
            return
        self.strokes.pop()
        self._redraw_from_strokes()
        self._update_preview()

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.strokes.clear()
        self._last_point = None
        self._update_preview()

    def _select_digit(self, digit: int):
        self.selected_digit.set(digit)
        self._highlight_selected_button()
        self._refresh_progress()

    def _highlight_selected_button(self):
        for i, btn in enumerate(self.digit_buttons):
            if i == self.selected_digit.get():
                btn.configure(bg=MAUVE, fg=BG)
            else:
                btn.configure(bg=BG_LIGHT, fg=TEXT)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _redraw_from_strokes(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        r = self._brush_r()
        for stroke in self.strokes:
            for i, (x, y) in enumerate(stroke):
                if i == 0:
                    self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                            fill="white", outline="white")
                    self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
                else:
                    px, py = stroke[i - 1]
                    self.canvas.create_line(px, py, x, y, fill="white",
                                            width=r * 2, capstyle="round", joinstyle="round")
                    self.pil_draw.line([(px, py), (x, y)], fill=255, width=r * 2)
                    self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _crop_and_resize(self, img: Image.Image) -> Image.Image:
        bbox = img.getbbox()
        if bbox is None:
            return img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

        margin = max(10, int(self.canvas_size * 0.05))
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(img.width, x1 + margin)
        y1 = min(img.height, y1 + margin)

        cropped = img.crop((x0, y0, x1, y1))

        w, h = cropped.size
        size = max(w, h)
        padded = Image.new("L", (size, size), 0)
        padded.paste(cropped, ((size - w) // 2, (size - h) // 2))

        return padded.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    def _update_preview(self):
        resized = self._crop_and_resize(self.pil_image)
        display = resized.resize((128, 128), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(display)
        self.preview_label.configure(image=self._preview_photo)

    def _count_images(self, digit: int) -> int:
        folder = os.path.join(OUTPUT_DIR, str(digit))
        if not os.path.isdir(folder):
            return 0
        return len([f for f in os.listdir(folder)
                    if f.startswith(f"{digit}-") and f.endswith(".png")])

    def _refresh_progress(self):
        total = 0
        for d in range(10):
            count = self._count_images(d)
            total += count
            bar_filled = min(count, TARGET_PER_DIGIT)
            bar = "█" * bar_filled + "░" * (TARGET_PER_DIGIT - bar_filled)
            status = "✅" if count >= TARGET_PER_DIGIT else f"{count}/{TARGET_PER_DIGIT}"
            self.progress_labels[d].configure(
                text=f" [{bar}] {status}",
                fg=GREEN if count >= TARGET_PER_DIGIT else YELLOW,
            )

        complete_text = "  🎉 Complete!" if total >= 10 * TARGET_PER_DIGIT else ""
        self.total_label.configure(
            text=f"Total: {total} / {10 * TARGET_PER_DIGIT}{complete_text}"
        )

        d = self.selected_digit.get()
        count = self._count_images(d)
        self.digit_info_var.set(
            f"Drawing:  {THAI_DIGITS[d]}  ({d})   —   {count}/{TARGET_PER_DIGIT}"
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    root = tk.Tk()
    ThaiDigitCollector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
