import cv2
import numpy as np


def remove_raindrops(input_image_path, output_image_path, mask_output_path=None):
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {input_image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    cv2.imwrite(output_image_path, restored)
    if mask_output_path:
        cv2.imwrite(mask_output_path, mask)
    return restored, mask


if __name__ == "__main__":
    remove_raindrops("data/rainy_scene.png", "results/rain_removed.png", "results/rain_mask.png")
