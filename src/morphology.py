import numpy as np


def binary2set(img):
    img = np.asarray(img)
    r, c = np.nonzero(img)
    return set(zip(r, c))


def set2binary(points, shape):
    h, w = shape
    result = np.zeros((h, w), dtype=np.uint8)
    for r, c in points:
        if 0 <= r < h and 0 <= c < w:
            result[r, c] = 1
    return result


def get_offsets(se, origin):
    orow, ocol = origin
    return [(r - orow, c - ocol) for r, c in se]


def erode(img, img_shape, se, origin):
    h, w = img_shape
    offsets = get_offsets(se, origin)
    result = set()

    for r in range(h):
        for c in range(w):
            if all((r + dr, c + dc) in img for dr, dc in offsets):
                result.add((r, c))
    return result


def dilate(img, img_shape, se, origin):
    h, w = img_shape
    offsets = get_offsets(se, origin)
    result = set()

    for r, c in img:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result.add((nr, nc))
    return result


def opening(img, img_shape, se, origin):
    return dilate(erode(img, img_shape, se, origin), img_shape, se, origin)


def closing(img, img_shape, se, origin):
    return erode(dilate(img, img_shape, se, origin), img_shape, se, origin)


def structuring_element_square(size=3):
    se = np.ones((size, size), dtype=np.uint8)
    return binary2set(se), (size // 2, size // 2)


def structuring_element_cross(size=3):
    se = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    se[mid, :] = 1
    se[:, mid] = 1
    return binary2set(se), (mid, mid)
