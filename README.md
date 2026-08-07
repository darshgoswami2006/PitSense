# PitSense — Pothole Detection & Speed Advisory System

A computer vision system that detects potholes in road footage, estimates their depth, classifies severity, and recommends safe driving speeds — all through a desktop GUI.

---

## Demo

<img src="https://github.com/user-attachments/assets/b998ca2b-26bc-4848-97a2-fa15aecba3c1" alt="PitSense GUI" width="872" />
<img src="https://github.com/user-attachments/assets/3f32ca11-4acd-4c46-8142-adc15f38fae2" alt="PitSense Output" width="1627" />

---

## Features

- **Pothole Detection** — YOLOv8s model fine-tuned on a merged dataset of 7,739 Indian and international road images. Detects single and multiple adjacent potholes per frame using Non-Maximum Suppression (NMS)
- **Depth Estimation** — MiDaS monocular depth model estimates relative pothole depth from a single camera, no LiDAR or stereo camera required
- **Temporal Smoothing** — ByteTrack multi-object tracker assigns persistent IDs to each pothole across frames; depth scores are averaged over a 10-frame rolling window to eliminate flickering
- **Severity Classification** — each detection classified as LOW, MEDIUM, or HIGH based on depth score and bounding box area
- **Physics-Based Speed Advisory** — safe crossing speed calculated using an exponential decay formula derived from pothole depth and size, not hardcoded thresholds
- **Two-Layer Speed Detection** — OCR reads GPS speed overlays burned into dashcam footage; optical flow estimation is used as a fallback when no overlay is present
- **Desktop GUI** — no terminal needed; browse video or image files, track progress live, outputs saved with timestamps so nothing is ever overwritten
- **Image and Video support** — accepts `.mp4 .avi .mov .mkv .webm` for video and `.jpg .jpeg .png .bmp .tiff .webp` for images

---

## How It Works

```
Input (Video or Image)
        │
        ▼
YOLOv8s Detection ──► Bounding boxes + confidence scores
        │
        ▼
MiDaS Depth Estimation ──► Relative depth map per frame
        │
        ▼
ByteTrack Tracking ──► Persistent pothole IDs across frames
        │
        ▼
Depth Smoothing ──► 10-frame moving average per track ID
        │
        ▼
Severity Classifier ──► LOW / MEDIUM / HIGH per detection
        │
        ├── OCR Speed Reader (primary) ──► reads GPS overlay from dashcam
        │         │
        └── Optical Flow (fallback) ──► Farneback algorithm on road region
                  │
                  ▼
        Speed Band: SLOW / MODERATE / FAST
                  │
                  ▼
Physics Advisory Engine ──► v_safe = 40 × e^(−3.5d) × size_penalty
                  │
                  ▼
        Annotated Output (Video / Image)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8s (Ultralytics) |
| Depth Estimation | MiDaS v2.1 Small |
| Multi-Object Tracking | ByteTrack (built into Ultralytics) |
| Speed Detection | EasyOCR + Farneback Optical Flow |
| GUI | Python Tkinter |
| Video Processing | OpenCV |
| Deep Learning | PyTorch + CUDA |
| Training Data | BharatPothole + Public Pothole (merged) |

---

## Model Performance

Three versions were trained during development. All evaluated on the same validation set:

| Metric | v1 (YOLOv8n, 50ep) | v2 (YOLOv8n, 100ep) | v3 (YOLOv8s, 100ep) |
|---|---|---|---|
| mAP50 | 0.467 | 0.505 | **0.606** |
| mAP50-95 | 0.180 | 0.213 | **0.292** |
| Precision | 0.578 | 0.625 | **0.728** |
| Recall | 0.437 | 0.503 | **0.560** |
| Parameters | 3M | 3M | **11M** |
| Train Time | 1.7 hr | 7.1 hr | 5.7 hr |

v3 achieves a **30% improvement in mAP50** over the baseline.

---

## Speed Advisory Logic

Safe crossing speed is calculated per pothole using an exponential decay formula:

```
v_safe = 40 × e^(−3.5 × depth_score) × (1 − min(8 × area_ratio, 0.35))
```

| Depth Score | Estimated Safe Speed |
|---|---|
| 0.10 | ~28 km/h |
| 0.20 | ~20 km/h |
| 0.30 | ~14 km/h |
| 0.50 | ~7 → 5 km/h |

When multiple potholes appear in the same frame, the worst-case speed is further reduced by 15%.

Urgency levels:

| Speed reduction needed | Banner message |
|---|---|
| > 35 km/h | `!! BRAKE NOW — Reduce to X km/h` |
| > 15 km/h | `>> Reduce speed to X km/h` |
| ≤ 15 km/h | `> Ease to X km/h` |
| None needed | `OK Safe to proceed` |

---

## Severity Classification

| Severity | Condition |
|---|---|
| HIGH | Depth score > 0.30 **or** pothole area > 4% of frame |
| MEDIUM | Depth score > 0.15 **or** pothole area > 1.5% of frame |
| LOW | Otherwise |

---

## Vehicle Type Profiles

The GUI lets you select your vehicle type before processing. Each profile has its own optical flow thresholds and speed band estimates:

| Vehicle | FAST threshold | MODERATE threshold | Est. FAST speed |
|---|---|---|---|
| Bicycle | 0.6 | 0.25 | ~25 km/h |
| Motorcycle | 1.2 | 0.5 | ~60 km/h |
| Car / SUV | 2.0 | 0.8 | ~80 km/h |
| Bus / Truck | 1.5 | 0.6 | ~60 km/h |

---

## Dataset

Trained on a merged dataset combining two sources:

| Dataset | Images | Format | Source |
|---|---|---|---|
| BharatPothole | ~7,074 | YOLO | Roboflow Universe (dashcam, Indian roads) |
| Roboflow Public Pothole | 665 | VOC → YOLO | public.roboflow.com |
| **Merged Total** | **7,739** | YOLO | train: 5,599 / valid: 1,411 / test: 729 |

Both datasets are licensed CC BY 4.0.

---

## Installation

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA (recommended)

### 1. Clone the repository
```bash
git clone https://github.com/LoneRead/PitSense.git
cd PitSense
```

### 2. Create a virtual environment with Python 3.12
```bash
py -3.12 -m venv pothole_env
pothole_env\Scripts\activate
```

### 3. Install PyTorch with CUDA

**For most NVIDIA GPUs (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For RTX 50 series / Blackwell GPUs (CUDA 12.8):**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Install remaining dependencies
```bash
pip install ultralytics opencv-python timm numpy matplotlib easyocr
```

### 5. Download model weights

The trained weights (`best.pt`) are not included in the repo due to file size. Either:
- Train your own using `train.py` (see below)
- Download the pretrained weights from [Releases](https://github.com/LoneRead/PitSense/releases)

Place weights at:
```
runs/pothole_v3/weights/best.pt
```

---

## Usage

### Run the GUI
```bash
python app.py
```

1. Select your **Vehicle Type** from the dropdown
2. Click **Browse Video** or **Browse Image** to select your file
3. Click **Run PitSense**
4. Watch the processing log and progress bar
5. Click **Open Output Folder** when done

Output files are saved as:
- Video: `videoname_pitsense_YYYYMMDD_HHMMSS.mp4`
- Image: `imagename_pitsense_YYYYMMDD_HHMMSS.png`

### Merge datasets
```bash
python merge_datasets.py
```

Converts the archive (Pascal VOC) dataset to YOLO format and merges it with BharatPothole into `merged_dataset/`.

### Train your own model
```bash
python train.py
```

Requires the merged dataset at `merged_dataset/data.yaml`. Trains YOLOv8s for 100 epochs and saves to `runs/pothole_v3/weights/best.pt`.

---

## Project Structure

```
PitSense/
├── app.py               # Main GUI application + full pipeline
├── depth.py             # MiDaS depth estimation wrapper
├── train.py             # YOLOv8s training script
├── merge_datasets.py    # Dataset conversion and merging script
├── merged_dataset/
│   └── data.yaml        # Unified dataset config
├── runs/
│   └── pothole_v3/
│       └── weights/
│           └── best.pt  # Trained model weights (not in repo)
├── outputs/             # Processed output files (not in repo)
├── .gitignore
└── README.md
```

---

## Key Formulas

| Formula | Purpose |
|---|---|
| `confidence = P(obj) × IoU` | YOLO detection score |
| `IoU = Overlap / Union` | Box overlap for NMS |
| `d = \|μ_pit − μ_road\| / μ_road` | Relative depth score (0–1) |
| `d̄ = (1/N) Σ dₖ` | Temporal depth smoothing |
| `v_safe = 40·e^(−3.5d) × penalty` | Physics-based safe crossing speed |
| `M = mean(√(fx²+fy²))` | Optical flow magnitude |
| `out = α·overlay + (1−α)·frame` | Semi-transparent overlay blend |

---

## Limitations

- Depth estimation is **relative**, not absolute — the system cannot output precise centimetre measurements without camera calibration
- Optical flow speed estimation is less reliable on open highways where there are few close objects in frame — OCR from GPS overlay is preferred when available
- The model was trained primarily on Indian road footage and may generalise less well to other regions
- Processing runs offline only — not real-time on live camera feeds

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Intel MiDaS](https://github.com/isl-org/MiDaS)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BharatPothole Dataset](https://universe.roboflow.com/yolo-ewrwa/dashcam-mg6en/dataset/14)
- [Roboflow Public Pothole Dataset](https://public.roboflow.com/object-detection/pothole)

---

*Built as part of an internship project — June / July 2026*
