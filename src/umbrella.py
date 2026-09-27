import os
import numpy as np
from PIL import Image
import matplotlib.colors as colors
from scipy import ndimage

COLORS = {
    "green": 120 / 360,
    "blue": 240 / 360,
    "cyan": 180 / 360,
    "magenta": 300 / 360,
    "yellow": 60 / 360,
}


def recolor_umbrella(input_path, output_dir="results/umbrella"):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    hsv = colors.rgb_to_hsv(img)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    red = ((h < 0.06) | (h > 0.93)) & (s > 0.25) & (v > 0.15)
    se = np.ones((3, 3))
    red = ndimage.binary_opening(red, structure=se, iterations=2)
    red = ndimage.binary_closing(red, structure=se, iterations=2)

    labels, count = ndimage.label(red)
    area = ndimage.sum(red, labels, range(1, count + 1))
    centroids = ndimage.center_of_mass(red, labels, range(1, count + 1))
    if count == 0:
        raise ValueError("No red region detected.")

    largest_area = area.max()
    H, W = h.shape
    area_thresh = 0.01 * largest_area
    x_lo, x_hi = 0.25 * W, 0.75 * W
    wanted = [
        i + 1 for i in range(count)
        if area[i] >= area_thresh and x_lo <= centroids[i][1] <= x_hi
    ]
    mask = np.isin(labels, wanted)
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=2)
    alpha = np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.5), 0, 1)

    outputs = {}
    for colour, target_hue in COLORS.items():
        new_hsv = np.stack([np.full_like(h, target_hue), s, v], axis=2)
        new_img = colors.hsv_to_rgb(new_hsv)
        result = img * (1 - alpha[:, :, None]) + new_img * alpha[:, :, None]
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(output_dir, f"umbrella_{colour}.png")
        Image.fromarray(result).save(path)
        outputs[colour] = path
    return outputs


if __name__ == "__main__":
    print(recolor_umbrella("data/umbrella.png"))
