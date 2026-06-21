# PitSense — Pothole Detection & Speed Advisory System

A computer vision system that detects potholes in road footage, estimates their depth, classifies severity, and recommends safe driving speeds — all through a desktop GUI.

---

## Features

- **Pothole Detection** — YOLOv8 model fine-tuned on Indian dashcam road footage, detects single and multiple adjacent potholes per frame
- **Depth Estimation** — MiDaS monocular depth model estimates relative pothole depth from a single camera
- **Severity Classification** — each detection classified as LOW, MEDIUM, or HIGH based on depth score and bounding box area
- **Speed Advisory** — real-time recommendation overlay (Safe / Reduce to 30 km/h / Slow to 10 km/h)
- **Desktop GUI** — no terminal needed; browse any video file, track progress live, outputs saved with timestamps so nothing is overwritten

---

## How It Works

```
Input Video
    │
    ▼
YOLOv8 Detection ──► Bounding boxes + confidence scores
    │
    ▼
MiDaS Depth Estimation ──► Depth map per frame
    │
    ▼
Severity Classifier ──► LOW / MEDIUM / HIGH per detection
    │
    ▼
Speed Advisory Engine ──► Annotated output video
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Depth Estimation | MiDaS v2.1 Small |
| GUI | Python Tkinter |
| Video Processing | OpenCV |
| Deep Learning | PyTorch + CUDA |
| Training Data | BharatPothole Dataset (Indian roads) |

---

## Model Performance

Trained for 100 epochs on the BharatPothole dataset (dashcam footage from Indian roads):

| Metric | Score |
|---|---|
| mAP50 | 0.505 |
| mAP50-95 | 0.213 |
| Precision | 0.625 |
| Recall | 0.503 |

---

## Installation

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA (recommended)

### Setup

**1. Clone the repository:**
```bash
git clone https://github.com/LoneRead/PitSense.git
cd PitSense
```

**2. Create a virtual environment:**
```bash
py -3.12 -m venv pothole_env
pothole_env\Scripts\activate
```

**3. Install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
> For RTX 50 series GPUs (Blackwell), use `cu128` nightly instead:
> `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

**4. Install remaining dependencies:**
```bash
pip install ultralytics opencv-python timm numpy matplotlib
```

**5. Download the trained model weights:**

The model weights (`best.pt`) are not included in this repo due to file size. You can either:
- Train your own using `train.py` (see below)
- Download the pretrained weights from [Releases](https://github.com/LoneRead/PitSense/releases)

Place the weights at:
```
runs/pothole_v2/weights/best.pt
```

---

## Usage

### Run the GUI app
```bash
python app.py
```

1. Click **Browse Video** to select a road video (`.mp4`, `.avi`, `.mov`, `.mkv`)
2. Click **Run PitSense**
3. Watch the processing log and progress bar
4. Click **Open Output Folder** when done — output is saved as `videoname_pitsense_TIMESTAMP.mp4`

### Train your own model
```bash
python train.py
```

Requires the BharatPothole dataset placed at:
```
BharatPotHole/BharatPotHole/BharatPotHole/data.yaml
```

---

## Project Structure

```
PitSense/
├── app.py          # Main GUI application + full pipeline
├── depth.py        # MiDaS depth estimation wrapper
├── train.py        # YOLOv8 training script
├── .gitignore
└── README.md
```

---

## Severity & Speed Advisory Logic

| Severity | Condition | Advisory |
|---|---|---|
| LOW | Depth < 0.15 and area < 1.5% of frame | Safe to proceed |
| MEDIUM | Depth 0.15–0.30 or area 1.5–4% of frame | Reduce to 30 km/h |
| HIGH | Depth > 0.30 or area > 4% of frame | Slow to 10 km/h |

---

## Dataset

Trained on the **BharatPothole** dataset — dashcam footage collected from Indian roads, annotated for pothole detection in YOLO format.

- Source: [Roboflow Universe](https://universe.roboflow.com/yolo-ewrwa/dashcam-mg6en/dataset/14)
- License: CC BY 4.0
- Classes: 1 (`pothole`)

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Intel MiDaS](https://github.com/isl-org/MiDaS)
- [BharatPothole Dataset](https://universe.roboflow.com/yolo-ewrwa/dashcam-mg6en)

