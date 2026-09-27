import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from skimage import data

from src.bit_planes import reconstruct_msb_to_lsb
from src.connect_dots import connect_dots
from src.dot_reconstruction import recreate_dot_pattern
from src.morphology import binary2set, set2binary, structuring_element_square, structuring_element_cross, erode, dilate, opening, closing
from src.object_removal import remove_leaves_and_twigs
from src.rain_removal import remove_raindrops
from src.restoration import run_experiments
from src.umbrella import recolor_umbrella


def morphology_demo():
    import numpy as np
    img = np.zeros((24, 24), dtype=np.uint8)
    img[4:18, 4:18] = 1
    img[9:13, 9:13] = 0
    img[1:3, 1:3] = 1
    img[17, 3] = 1
    img[18, 4] = 1
    points = binary2set(img)
    for name, (se, origin) in {
        "square": structuring_element_square(3),
        "cross": structuring_element_cross(3),
    }.items():
        for op_name, op in [("erosion", erode), ("dilation", dilate), ("opening", opening), ("closing", closing)]:
            result = set2binary(op(points, img.shape, se, origin), img.shape)
            from PIL import Image
            Image.fromarray(result * 255).save(f"results/morphology/{name}_{op_name}.png")


def main():
    import os
    os.makedirs("results/morphology", exist_ok=True)
    recreate_dot_pattern()
    connect_dots()
    morphology_demo()
    run_experiments()
    reconstruct_msb_to_lsb(data.camera())
    recolor_umbrella("data/umbrella.png")
    remove_leaves_and_twigs("data/road_image.png", "results/road_cleaned.png", "results/road_mask.png")
    remove_raindrops("data/rainy_scene.png", "results/rain_removed.png", "results/rain_mask.png")
    print("All demos completed.")


if __name__ == "__main__":
    main()
