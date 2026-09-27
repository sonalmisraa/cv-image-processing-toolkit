import csv
import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage import color, data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def add_gaussian_noise(img, sigma=25, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.05, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    noisy = img.copy()
    n_pixels = img.size
    n_salt = int(amount * n_pixels * 0.5)
    n_pepper = int(amount * n_pixels * 0.5)
    coords = tuple(rng.integers(0, d, n_salt) for d in img.shape)
    noisy[coords] = 255
    coords = tuple(rng.integers(0, d, n_pepper) for d in img.shape)
    noisy[coords] = 0
    return noisy


def add_poisson_noise(img, scale_factor=25.0, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    scaled = img.astype(np.float64) * scale_factor / 255.0
    noisy = rng.poisson(scaled) * (255.0 / scale_factor)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_speckle_noise(img, var=0.05, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    noise = rng.normal(0, var ** 0.5, img.shape)
    noisy = img.astype(np.float64) + img.astype(np.float64) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def restore_gaussian(img, ksize=5, sigma=1.5):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def restore_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def restore_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def restore_lee(img, win=7, cu=0.05):
    img_f = img.astype(np.float64)
    mean_local = uniform_filter(img_f, size=win)
    sq_mean_local = uniform_filter(img_f ** 2, size=win)
    var_local = np.maximum(sq_mean_local - mean_local ** 2, 0)
    ci = var_local / (mean_local ** 2 + 1e-8)
    weight = np.clip(1 - (cu / (ci + 1e-8)), 0, 1)
    restored = mean_local + weight * (img_f - mean_local)
    return np.clip(restored, 0, 255).astype(np.uint8)


def run_experiments(output_dir="results/restoration"):
    os.makedirs(output_dir, exist_ok=True)
    clean = color.rgb2gray(data.astronaut())
    clean = (clean * 255).astype(np.uint8)
    rng = np.random.default_rng(42)

    experiments = [
        ("Gaussian Noise", "gaussian", lambda: add_gaussian_noise(clean, 25, rng), restore_gaussian),
        ("Salt-and-Pepper Noise", "salt_pepper", lambda: add_salt_pepper_noise(clean, 0.05, rng), restore_median),
        ("Poisson Noise", "poisson", lambda: add_poisson_noise(clean, 25.0, rng), restore_bilateral),
        ("Speckle Noise", "speckle", lambda: add_speckle_noise(clean, 0.05, rng), restore_lee),
    ]

    rows = []
    for name, key, noise_fn, restore_fn in experiments:
        noisy = noise_fn()
        restored = restore_fn(noisy)
        rows.append({
            "Noise Model": name,
            "PSNR Noisy (dB)": round(psnr(clean, noisy, data_range=255), 2),
            "SSIM Noisy": round(ssim(clean, noisy, data_range=255), 4),
            "PSNR Restored (dB)": round(psnr(clean, restored, data_range=255), 2),
            "SSIM Restored": round(ssim(clean, restored, data_range=255), 4),
        })
        cv2.imwrite(os.path.join(output_dir, f"{key}_noisy.png"), noisy)
        cv2.imwrite(os.path.join(output_dir, f"{key}_restored.png"), restored)

    cv2.imwrite(os.path.join(output_dir, "clean_original.png"), clean)
    with open(os.path.join(output_dir, "summary_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    for row in run_experiments():
        print(row)
