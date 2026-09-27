# Methodology Notes

## Morphology

Binary images and structuring elements are represented as sets of foreground coordinates. A structuring element is converted to offsets relative to its origin. Erosion checks whether every offset is present at a candidate location; dilation adds every offset to each foreground pixel. Opening is erosion followed by dilation, while closing is dilation followed by erosion.

## Restoration

Four degradation/restoration pairs are evaluated:

| Degradation | Restoration |
|---|---|
| Gaussian | Gaussian filter |
| Salt-and-pepper | Median filter |
| Poisson | Bilateral filter |
| Speckle | Lee filter |

The clean, degraded and restored images are compared using PSNR and SSIM.

## Rain removal

The grayscale image is processed with a 9×9 elliptical morphological top-hat transform. Bright thin structures are thresholded to form a rain mask, the mask is dilated, and only the detected regions are restored with Telea inpainting.

## Leaf/twig removal

Leaves are detected from HSV/LAB color characteristics and refined with morphology and connected-component filtering. Thin twigs are detected using color/intensity cues and rotated line-shaped morphological kernels. The combined mask is repaired using nearby road texture selected by local grayscale mean-squared error, followed by boundary-only smoothing.

## Umbrella recoloring

The image is converted to HSV. Red pixels are selected from both ends of the hue range, the mask is cleaned morphologically, connected components are filtered by area and centroid location, and the selected region is recolored by changing hue while preserving saturation and value.
