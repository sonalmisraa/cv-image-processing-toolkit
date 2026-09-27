# Computer Vision & Image Processing Toolkit

A collection of classical computer-vision and image-processing pipelines developed in Python, covering image reconstruction, morphological processing, denoising, segmentation, object removal, color manipulation, and quantitative restoration evaluation.

The project consolidates work completed for **EE604: Image Processing** into a reusable repository rather than separate coursework submissions.

## What is implemented?

### 1. Image restoration and denoising
- Gaussian noise + Gaussian filtering
- Salt-and-pepper noise + median filtering
- Poisson noise + bilateral filtering
- Speckle noise + Lee filtering
- PSNR and SSIM evaluation before and after restoration

### 2. Binary morphology from scratch
- Binary image ↔ set-of-coordinates conversion
- Erosion
- Dilation
- Opening
- Closing
- Configurable structuring-element origin
- Square and cross structuring elements

The core morphology implementation does not rely on built-in morphology functions.

### 3. Segmentation and object manipulation
- HSV/LAB color-based segmentation
- Connected-component analysis
- Umbrella recoloring while preserving saturation/value information
- Leaf and twig detection
- Morphological mask refinement

### 4. Image reconstruction and restoration
- Dot-pattern generation from a mathematical intensity field
- Ordered dot connection and binary reconstruction
- Rain-streak detection using a morphological top-hat transform
- Targeted restoration using Telea inpainting
- Texture-based road reconstruction for detected leaves/twigs

### 5. Bit-plane analysis
- Extraction of all 8 bit planes
- MSB-to-LSB progressive reconstruction
- Exact reconstruction verification
- MP4 generation of the reconstruction sequence

## Repository structure

```text
cv-image-processing-toolkit/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── road_image.png
│   ├── rainy_scene.png
│   └── umbrella.png
├── src/
│   ├── morphology.py
│   ├── restoration.py
│   ├── bit_planes.py
│   ├── dot_reconstruction.py
│   ├── connect_dots.py
│   ├── umbrella.py
│   ├── object_removal.py
│   └── rain_removal.py
├── scripts/
│   └── run_demos.py
└── results/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run individual modules

### Restoration

```bash
python -m src.restoration
```

Outputs are written to `results/restoration/`, including `summary_metrics.csv`.

### Bit-plane reconstruction

```bash
python -m src.bit_planes
```

### Dot-pattern reconstruction

```bash
python -m src.dot_reconstruction
python -m src.connect_dots
```

### Umbrella recoloring

```bash
python -m src.umbrella
```

### Leaf/twig removal

```bash
python -m src.object_removal
```

### Rain removal

```bash
python -m src.rain_removal
```

### Run the full demo suite

```bash
python scripts/run_demos.py
```

## Restoration evaluation

The restoration experiment reports:

- **PSNR (Peak Signal-to-Noise Ratio)** in dB
- **SSIM (Structural Similarity Index)**

The comparison is made against the original clean grayscale image for each noise model.

## Key computer-vision techniques

| Area | Techniques |
|---|---|
| Image restoration | Gaussian, median, bilateral, Lee filtering |
| Evaluation | PSNR, SSIM, MSE |
| Morphology | Erosion, dilation, opening, closing, top-hat, gradient |
| Segmentation | HSV, LAB, thresholding |
| Object analysis | Connected components, centroids, shape filtering |
| Reconstruction | Texture matching, patch replacement, inpainting |
| Representation | Bit-plane slicing |

## Notes

This repository is a cleaned, reusable organization of work originally developed for EE604 Image Processing coursework. The code has been separated into modules so that the individual algorithms can be run independently and reused on other images.
