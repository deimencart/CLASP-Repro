# CLASP: Cloth Assistance with Semantic Keypoints & Interactive Perception

> **Independent Academic Reproduction & Experimental Implementation**  
> *Reference: CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints (Deng et al., 2025)*  
> *Stack: PyTorch, DINOv2 ViT, Inverse Pinhole 3D, Closed-Loop FSM Control*

---

> [!IMPORTANT]
> **Academic Reproduction & Non-Plagiarism Disclaimer**  
> This repository contains an **independent academic reproduction and experimental implementation** created for educational and robotics research purposes. All foundational concepts, problem formulations, and architectural designs belong and are fully attributed to their original authors (*Deng et al., 2025*). This implementation is non-commercial and carries no intent of plagiarism.

---

## 📌 Problem Statement

In robotic garment manipulation, locating reliable grasp points on crumpled, folded, or unstructured clothing typically requires training task-specific supervised detectors. These supervised networks tend to overfit and fail whenever fabric texture, color, or deformation changes.

**CLASP** addresses this challenge through two core ideas:
1. **Zero-Shot Semantic Transfer:** Leveraging a frozen vision foundation model (**DINOv2**) that intrinsically understands semantic parts across instances. By annotating key landmarks on a single canonical prototype (e.g., waistband, pockets, hem, crotch), the algorithm transfers these grasp targets to unseen garments without any neural network retraining.
2. **Closed-Loop Interactive Perception:** Vision alone is insufficient because deformable fabrics slip and multi-layer occlusions easily lead to accidental multi-layer grasps. The robot gripper uses tactile touchdown force feedback and proprioceptive jaw width checks to verify and stabilize the grasp before lifting.

---

## 🛠️ Implementation Overview

The implemented pipeline connects raw RGB-D observations directly to robot Cartesian execution:

```
[ Annotated Canonical Prototype ]       [ Observed Real Garment (RGB-D) ]
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                 [ Frozen DINOv2 ViT-B/14 Backbone ]
                   (Dense descriptors extracted on GPU)
                                 |
                                 v
                 [ Coarse-to-Fine Semantic Matcher ]
                   (Cosine similarity landmark transfer)
                                 |
                                 v
                 [ Inverse 3D Pinhole Deprojection ]
                   (Camera to Robot Base Frame SE(3))
                                 |
                                 v
                 [ Closed-Loop Interactive FSM ]
                   1. Approach & Standoff Hover (+6 cm)
                   2. Tactile Touchdown (F_z >= 3.0 N)
                   3. Proprioceptive Jaw Gap Verification
                   4. Lift-and-Verify (+15 cm tension check)
```

### 1. Semantic Extraction & Matching
* **DINOv2 Extractor (`src/02_dino_extractor.py`):** Runs images through a frozen ViT-B/14 backbone and normalizes dense feature vectors per pixel in GPU memory.
* **Keypoint Matcher (`src/03_matcher.py`):** Computes cosine similarity between observed descriptors and prototype landmarks to locate semantic targets.
* **In-Plane Rotation Aligner (`src/04_rotation_aligner.py`):** Aligns the principal 2D axis of the garment to improve correspondence robustness under arbitrary rotations.

### 2. 3D Deprojection to Robot Base Frame
* **3D Deprojector (`src/05_deprojector_3d.py`):** Converts 2D pixel coordinates $(u, v)$ and metric depth $Z$ into 3D camera coordinates, applying hand-eye calibration extrinsics $T_{\text{base}}^{\text{cam}}$ to output target Cartesian positions in meters.

### 3. Closed-Loop Grasp State Machine
* **Interactive Perception (`src/clasp_interactive_perception.py`):**
  * **Hover:** Moves the end-effector to a standoff position $+6\,\text{cm}$ above the target along the cloth surface normal.
  * **Tactile Touchdown:** Lowers the gripper at a controlled speed ($1.5\,\text{cm/s}$) until registering a normal contact force of $F_z \ge 3.0\,\text{N}$, preventing hard impacts against the table.
  * **Proprioceptive Verification:** Closes fingers and inspects the residual jaw gap $\delta_{\text{grip}}$ to ensure fabric is captured while detecting false closures or overly thick hems.
  * **Lift & Verify:** Raises the gripper $+15\,\text{cm}$ vertically to confirm grasp retention under gravity.

---

## 📊 Visual Evidence & Matching Results

The system reliably transfers semantic landmarks across diverse real-world garments (jeans, chinos, and sweatpants):

| Real Jeans (Sample 2) | Real Jeans (Sample 3) |
| :---: | :---: |
| ![Matching Jeans 2](docs/images/match_real_jeans_2.png) | ![Matching Jeans 3](docs/images/match_real_jeans_3.png) |

| Cross-Category Transfer (Chino) | Cross-Category Transfer (Sweatpants) |
| :---: | :---: |
| ![Matching Chino](docs/images/match_trousers_chino_beige.png) | ![Matching Sweatpants](docs/images/match_trousers_sweatpants_grey.png) |

---

## 📁 Repository Structure

```text
clasp_semantic_keypoints/
├── README.md                                 <- Master documentation
├── .gitignore                                <- Git ignore rules
├── docs/
│   ├── CLASP_SemanticKeypoints_2025.pdf      <- Reference paper
│   ├── 01_CLASP_Framework_and_First_Principles.md
│   ├── 02_CLASP_Foundation_Models_Ecosystem.md
│   └── images/                               <- Matching visualizations
├── src/
│   ├── 02_dino_extractor.py                  <- DINOv2 GPU feature extractor
│   ├── 03_matcher.py                         <- Landmark correspondence matcher
│   ├── 04_rotation_aligner.py                <- 2D in-plane angle aligner
│   ├── 05_deprojector_3d.py                  <- Pinhole 2D to 3D base converter
│   ├── clasp_interactive_perception.py       <- Closed-loop FSM (Touchdown & Grasp)
│   ├── action_primitives.py                  <- High-level motion primitives
│   ├── inference.py                          <- End-to-end evaluation runner
│   ├── simulation_utils.py                   <- Simulation helpers
│   └── metric.py                             <- Landmark error evaluation
└── data/
    ├── prototypes/                           <- Annotated canonical prototypes
    └── samples/                              <- Evaluation garment images
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies
```bash
pip install torch torchvision numpy opencv-python scipy matplotlib
```

### 2. Run Semantic Keypoint Matching
Extract features and compute correspondence against the canonical prototype:
```bash
python src/02_dino_extractor.py
python src/03_matcher.py
```

### 3. Run Closed-Loop Interactive Perception
Execute the finite state machine with hover, force touchdown, and grasp verification:
```bash
python src/clasp_interactive_perception.py
```

---

## 📚 References
* Deng, et al. (2025). *CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints*. arXiv:2502.17983.
* Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR.
