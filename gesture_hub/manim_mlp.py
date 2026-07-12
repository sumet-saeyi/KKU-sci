from manim import *
import numpy as np
import json
import os
import cv2

class MNISTNetwork(Scene):
    def construct(self):
        # 1. Title
        title = Text("How the MLP Predicts your Drawing", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Load user data
        pred_val = 3
        if os.path.exists("prediction_data.json"):
            with open("prediction_data.json", "r") as f:
                data = json.load(f)
                pred_val = data.get("prediction", 3)
                
        # 2. Represent the Input Image as Pixels (0s and 1s)
        img_array = np.zeros((28, 28))
        if os.path.exists("user_digit.png"):
            img = cv2.imread("user_digit.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img_array = (img > 50).astype(int) # Threshold to 0 and 1
        else:
            # Fallback a rough "3"
            points = [(6, 20), (12, 22), (18, 18), (14, 12), (10, 12), (14, 10), (18, 6), (12, 2), (6, 6)]
            for p in points:
                cv2.circle(img_array, p, 2, 1, -1)
                
        # Build the 28x28 grid of text
        pixel_group = VGroup()
        for i in range(28):
            row_group = VGroup()
            for j in range(28):
                val = img_array[i, j]
                color = WHITE if val == 1 else DARK_GRAY
                # We use very small font size
                txt = Text(str(val), font_size=10, color=color)
                row_group.add(txt)
            row_group.arrange(RIGHT, buff=0.08)
            pixel_group.add(row_group)
        pixel_group.arrange(DOWN, buff=0.08)
        
        pixel_group.move_to(LEFT * 4)
        self.play(FadeIn(pixel_group), run_time=0.5)
        
        # 3. Create Neural Network Layers
        layer_sizes = [16, 8, 8, 10]
        layer_x = [-1, 1, 3, 5]
        
        layers = VGroup()
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            for j in range(size):
                y = (j - size / 2) * 0.4
                dot = Dot(point=np.array([layer_x[i], y, 0]), radius=0.08, color=WHITE)
                layer.add(dot)
            layers.add(layer)
            
        edges = VGroup()
        for i in range(len(layers) - 1):
            for dot1 in layers[i]:
                for dot2 in layers[i+1]:
                    edge = Line(dot1.get_center(), dot2.get_center(), stroke_width=0.3, stroke_opacity=0.3, color=GRAY)
                    edges.add(edge)
                    
        self.play(Create(edges), run_time=0.5)
        self.play(Create(layers), run_time=0.4)
        
        # 4. Animate Flattening
        flatten_text = Text("Flatten 28x28 -> 784 Pixels", font_size=20, color=WHITE).next_to(layers[0], UP).shift(LEFT)
        self.play(Write(flatten_text))
        
        # Condense the entire grid into the first layer
        self.play(
            pixel_group.animate.scale(0.05).move_to(layers[0].get_center()).set_opacity(0),
            run_time=0.5
        )
        self.play(layers[0].animate.set_color(WHITE))
        
        # 5. Forward Pass Animation
        def forward_pass(start_layer, end_layer, color=WHITE, time=0.4):
            animations = []
            for d1 in start_layer:
                for d2 in end_layer:
                    if np.random.random() > 0.8:
                        dot = Dot(d1.get_center(), radius=0.05, color=color)
                        animations.append(MoveAlongPath(dot, Line(d1.get_center(), d2.get_center())))
            if animations:
                self.play(*animations, run_time=time)
                
        h1_text = Text("Hidden Layer 1\n(512 Nodes)", font_size=18).next_to(layers[1], UP)
        self.play(Write(h1_text))
        forward_pass(layers[0], layers[1], color=WHITE, time=1.0)
        self.play(layers[1].animate.set_color(WHITE))
        
        h2_text = Text("Hidden Layer 2\n(512 Nodes)", font_size=18).next_to(layers[2], DOWN)
        self.play(Write(h2_text))
        forward_pass(layers[1], layers[2], color=WHITE, time=1.0)
        self.play(layers[2].animate.set_color(WHITE))
        
        out_text = Text("Output\n(10 Digits)", font_size=18).next_to(layers[3], UP)
        self.play(Write(out_text))
        forward_pass(layers[2], layers[3], color=WHITE, time=1.0)
        
        # 6. Final Prediction
        target_node = layers[3][9 - pred_val]
        
        self.play(
            layers[3].animate.set_color(DARK_GRAY),
            target_node.animate.set_color(GREEN).scale(2.0)
        )
        
        labels = VGroup()
        for i, dot in enumerate(layers[3]):
            val = 9 - i
            lbl = Text(str(val), font_size=20).next_to(dot, RIGHT)
            if val == pred_val:
                lbl.set_color(GREEN).scale(1.5)
            labels.add(lbl)
            
        self.play(Write(labels))
        
        result = Text(f"PREDICTION: {pred_val}", font_size=42, color=GREEN).next_to(labels, RIGHT).shift(UP*2)
        self.play(Write(result), run_time=0.5)
        
        self.wait(1)
