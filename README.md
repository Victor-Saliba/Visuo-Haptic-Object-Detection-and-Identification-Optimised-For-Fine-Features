# Hybrid Visuo-Haptic Object Detection and Identification Model (Optimised for Fine Features)

**University of Manchester student project**

## Overview

This project explores hybrid object detection and identification by combining traditional RGB-based vision with haptic sensing, enabling robust recognition of objects that require fine feature discrimination. While standard object detection frameworks like Faster R-CNN work well for many tasks, they often struggle with:

- Poor lighting conditions
- Features that are too fine to resolve with standard RGB images

To address this, we investigate two approaches for fine-feature object identification:

1. **Vision-based secondary classifier:** Crops of detected objects (from the original RGB frame) are passed to a ResNet-18 image classifier for further identification.
2. **Haptic classifier:** Gelsight sensor images of the object are used, allowing fine features (such as screw or nut size) to be detected even in challenging visual conditions.

### Case Study: Tool and Small Object Classification

A new superclass, `tools`, was added to the COCO dataset as a case study. The eight tool classes are:

- `screwdriver`
- `wrench`
- `pliers`
- `boxcutter`
- `stapler`
- `wire spool`
- `screws`
- `nuts`

Along with samples from the original 80 COCO classes, these make up the dataset.

#### Fine Feature Classification

The `screws` and `nuts` classes (called **small objects**) are further subdivided by metric size: `m2`, `m3`, `m4`, `m5`, `m6`, `m8`, `m9`, and `m10`. Classification of these fine-grained categories is performed by two secondary models:

- **Vision classifier:** ResNet-18 on cropped RGB images
- **Haptic classifier:** ResNet-18 on Gelsight sensor images

Our hypothesis was that the haptic model would outperform vision for fine features, and this was confirmed in our results.

*_(Insert model architecture image here if available)_*

---

## Repository Structure & Usage

- **MainUI.py**  
  The main demo script.  
  - Runs the hybrid system described above.
  - When a "small object" is detected, it:
    - Automatically classifies the crop using the vision ResNet-18 model.
    - Prompts you to bring the Gelsight sensor in contact with the object for haptic classification.
  - **Requirements:**  
    - You need both a camera and a Gelsight sensor to use this demo.
    - `MainUI.py` must be in the same folder as `for_models` and `gsrobotics`.

- **for_models/**  
  - Contains the fine-tuned Faster R-CNN model (with the 8 tool classes added to the original COCO-80).
  - Contains the secondary vision and Gelsight classifiers.
  - **Important:**  
    - Place the downloaded model resource file (from [this release link](https://github.com/Victor-Saliba/Hybrid-Visuo-Haptic-Object-Detection-and-Identification-Model-Optimised-For-Fine-Features/releases)) in this folder to run `MainUI.py`.

- **gsrobotics/**  
  - Contains required parts of the Gelsight SDK (needed for `MainUI.py`).

- **data/**  
  - Contains the datasets for training each respective model (as can be seen by the folder names).
  - **raw data** subfolder:  
    - Contains all images and Gelsight videos collected, organized by specific object names (e.g., "M8 Screw #2").
    - Gelsight database includes both video frames and still images.

---

## Models Provided

- **Fine-tuned Faster R-CNN**  
  For detection with the new tools superclass and the original COCO-80.
- **Vision Stage 2 Model**  
  ResNet-18 classifier for cropped images of "small objects."
- **GS Model**  
  ResNet-18 classifier for Gelsight (haptic) images.

---

## Installation & Requirements

Install dependencies from the provided `requirements.txt`:

```sh
pip install -r requirements.txt
```

**Note:**  
The requirements file includes everything needed for `MainUI.py` **except for OpenMMLab dependencies** (e.g., MMDetection, MMCV, etc.).  
- You must install OpenMMLab packages manually according to your CUDA version and system configuration.  
- See [OpenMMLab documentation](https://github.com/open-mmlab/mmdetection) for installation instructions.

---

## Fine-tuned Faster R-CNN Weights

- **The Faster R-CNN model weights are not included** in the git repo due to size limits.
- Download required file,`faster_rcnn_victor_coco88.pth`, from the [Releases page](https://github.com/Victor-Saliba/Hybrid-Visuo-Haptic-Object-Detection-and-Identification-Model-Optimised-For-Fine-Features/releases) and **place them in the `for_models` folder**.

---

## Quick Demo Instructions

1. Install dependencies and OpenMMLab manually.
2. Download `faster_rcnn_victor_coco88.pth` from [Releases page](https://github.com/Victor-Saliba/Hybrid-Visuo-Haptic-Object-Detection-and-Identification-Model-Optimised-For-Fine-Features/releases) and place in `for_models/`.
3. Ensure you have both a camera and a Gelsight sensor connected.
4. Run:
    ```sh
    python MainUI.py
    ```
5. Follow prompts for Gelsight sensing.

---

## Acknowledgments

- [COCO Dataset](https://cocodataset.org)
- [@open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- [@gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics)

---

## Contact

For questions, please open an issue or contact [Victor-Saliba](https://github.com/Victor-Saliba).
