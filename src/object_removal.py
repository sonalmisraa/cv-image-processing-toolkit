import cv2
import numpy as np


def get_leaf_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    _, _, B = cv2.split(lab)

    yellow = (H >= 7) & (H <= 42) & (S >= 80) & (V >= 55) & (B >= 155)
    mask = yellow.astype(np.uint8) * 255

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= 35 and max(width, height) >= 7:
            clean[labels == i] = 255

    return cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def get_twig_mask(image, leaf_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    brown = ((H <= 25) & (S >= 45) & (V <= 190)).astype(np.uint8) * 255
    local_background = cv2.GaussianBlur(gray, (0, 0), 7)
    darkness = local_background.astype(np.int16) - gray.astype(np.int16)
    dark_object = (darkness >= 8).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(brown, dark_object)
    candidate[leaf_mask > 0] = 0

    twig_response = np.zeros_like(candidate)
    for angle in range(0, 180, 10):
        kernel = np.zeros((21, 21), dtype=np.uint8)
        cv2.line(kernel, (2, 10), (18, 10), 1, 1)
        rotation = cv2.getRotationMatrix2D((10, 10), angle, 1.0)
        rotated_kernel = cv2.warpAffine(kernel, rotation, (21, 21), flags=cv2.INTER_NEAREST)
        response = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, rotated_kernel)
        twig_response = np.maximum(twig_response, response)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(twig_response, connectivity=8)
    clean_twigs = np.zeros_like(twig_response)
    for i in range(1, count):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        longest = max(width, height)
        shortest = max(1, min(width, height))
        if area >= 10 and longest >= 20 and longest / shortest >= 3:
            clean_twigs[labels == i] = 255

    return cv2.dilate(clean_twigs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def texture_fill(image, mask):
    result = image.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = mask.shape
    regions = list(range(1, count))
    regions.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    for component in regions:
        ys, xs = np.where(labels == component)
        if len(xs) < 30:
            continue
        x0, x1 = max(0, xs.min()-12), min(width-1, xs.max()+12)
        y0, y1 = max(0, ys.min()-12), min(height-1, ys.max()+12)
        patch_w, patch_h = x1-x0+1, y1-y0+1
        target = gray[y0:y1+1, x0:x1+1]
        component_mask = labels[y0:y1+1, x0:x1+1] == component
        context = ~component_mask
        if context.sum() < 40:
            continue

        cx, cy = (x0+x1)//2, (y0+y1)//2
        best_position, best_score = None, float("inf")
        for sy in range(max(0, cy-100), min(height-patch_h, cy+100)+1, 8):
            for sx in range(max(0, cx-100), min(width-patch_w, cx+100)+1, 8):
                source_mask = mask[sy:sy+patch_h, sx:sx+patch_w]
                if np.mean(source_mask > 0) > 0.02:
                    continue
                source = gray[sy:sy+patch_h, sx:sx+patch_w]
                score = np.mean((target[context] - source[context]) ** 2)
                if score < best_score:
                    best_score, best_position = score, (sx, sy)

        if best_position is None:
            continue
        sx, sy = best_position
        source_patch = result[sy:sy+patch_h, sx:sx+patch_w].copy()
        result[y0:y1+1, x0:x1+1][component_mask] = source_patch[component_mask]
        labels[labels == component] = 0

    return result


def remove_leaves_and_twigs(input_path, output_path, mask_path=None):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load {input_path}")
    leaf_mask = get_leaf_mask(image)
    twig_mask = get_twig_mask(image, leaf_mask)
    combined = cv2.bitwise_or(leaf_mask, twig_mask)
    combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cleaned = texture_fill(image, combined)
    blurred = cv2.GaussianBlur(cleaned, (3, 3), 0)
    boundary = cv2.morphologyEx(combined, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    cleaned[boundary > 0] = blurred[boundary > 0]

    cv2.imwrite(output_path, cleaned)
    if mask_path:
        cv2.imwrite(mask_path, combined)
    return cleaned, combined


if __name__ == "__main__":
    remove_leaves_and_twigs("data/road_image.png", "results/road_cleaned.png", "results/road_mask.png")
