import os
import cv2
import numpy as np
from skimage import data


def extract_bit_planes(img):
    return {b: ((img & (1 << (b - 1))) >> (b - 1)) for b in range(1, 9)}


def reconstruct_msb_to_lsb(img, output_dir="results/bit_planes", fps=1):
    os.makedirs(output_dir, exist_ok=True)
    planes = extract_bit_planes(img)
    order = [8, 7, 6, 5, 4, 3, 2, 1]
    recon = np.zeros_like(img, dtype=np.uint16)
    frames = []

    for b in range(1, 9):
        cv2.imwrite(os.path.join(output_dir, f"bitplane_{9-b:02d}_bit{b}_raw.png"), planes[b] * 255)

    for step, b in enumerate(order, start=1):
        recon += planes[b] * (1 << (b - 1))
        frame = np.clip(recon, 0, 255).astype(np.uint8)
        path = os.path.join(output_dir, f"recon_{step:02d}.png")
        cv2.imwrite(path, frame)
        frames.append(path)

    assert np.array_equal(frame, img)

    h, w = img.shape
    video_path = os.path.join(output_dir, "bitplane_reconstruction.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), False)
    for path in frames:
        writer.write(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    writer.release()
    return video_path


if __name__ == "__main__":
    reconstruct_msb_to_lsb(data.camera())
