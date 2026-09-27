import math
import os
from PIL import Image, ImageDraw


def recreate_dot_pattern(output_path="results/dot_pattern_recreated.png"):
    W, H = 1282, 1284
    S = 4
    canvas = Image.new("L", (W * S, H * S), 255)
    draw = ImageDraw.Draw(canvas)

    x0, y0 = 88, 58
    spacing = 31
    cols, rows = 36, 37
    cx, cy = 640, 642
    sigma_x, sigma_y = 238.0, 255.0
    r_max, r_min, gamma = 11.8, 0.45, 0.78

    for j in range(rows):
        y = y0 + j * spacing
        for i in range(cols):
            x = x0 + i * spacing
            q = ((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2
            darkness = math.exp(-0.5 * q) ** gamma
            radius = r_min + (r_max - r_min) * darkness
            if darkness < 0.025:
                radius *= darkness / 0.025
            r = radius * S
            xs, ys = x * S, y * S
            draw.ellipse((xs - r, ys - r, xs + r, ys + r), fill=28)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.resize((W, H), Image.Resampling.LANCZOS).save(output_path)
    return output_path


if __name__ == "__main__":
    print(recreate_dot_pattern())
