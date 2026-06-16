import cv2
import numpy as np
import torch
from ultralytics import YOLO
from depth import load_depth_model, estimate_depth, get_depth_score

# ── Configuration ────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Projects\runs\pothole_v1-5\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.35

# Severity thresholds
DEPTH_HIGH     = 0.3
DEPTH_MEDIUM   = 0.15
AREA_HIGH      = 0.04
AREA_MEDIUM    = 0.015

# Speed advisory
ADVISORY = {
    "HIGH":   {"slow": True,  "speed": 10,  "color": (0, 0, 255),   "msg": "SLOW DOWN to 10 km/h"},
    "MEDIUM": {"slow": True,  "speed": 30,  "color": (0, 165, 255), "msg": "Reduce speed to 30 km/h"},
    "LOW":    {"slow": False, "speed": None, "color": (0, 255, 0),  "msg": "Safe to proceed"},
}

# ── Severity Classifier ───────────────────────────────────────
def classify_severity(depth_score, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

    if depth_score > DEPTH_HIGH or area_ratio > AREA_HIGH:
        return "HIGH"
    elif depth_score > DEPTH_MEDIUM or area_ratio > AREA_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"

# ── Annotate Frame ────────────────────────────────────────────
def annotate_frame(frame, detections, depth_map):
    overall_severity = "LOW"
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    for det in detections:
        bbox = det["bbox"]
        conf = det["conf"]
        severity = det["severity"]
        advisory = ADVISORY[severity]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = advisory["color"]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with severity and confidence
        label = f"Pothole [{severity}] {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Track worst severity
        if severity_rank[severity] > severity_rank[overall_severity]:
            overall_severity = severity

    # Overall advisory banner at top
    adv = ADVISORY[overall_severity]
    banner_color = adv["color"]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), banner_color, -1)
    cv2.putText(frame, adv["msg"], (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Pothole count
    count_text = f"Potholes detected: {len(detections)}"
    cv2.putText(frame, count_text, (frame.shape[1] - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# ── Main Pipeline ─────────────────────────────────────────────
def run_pipeline(source):
    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH)

    print("Loading depth model...")
    depth_model, depth_transform = load_depth_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        return

    # Output video writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(
        r"C:\Projects\output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    frame_count = 0
    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []

        if results.boxes is not None and len(results.boxes):
            # Only run depth estimation if potholes detected (saves time)
            depth_map = estimate_depth(frame, depth_model, depth_transform)

            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                depth_score = get_depth_score(depth_map, bbox, frame.shape)
                severity = classify_severity(depth_score, bbox, frame.shape)

                detections.append({
                    "bbox": bbox,
                    "conf": conf,
                    "depth_score": depth_score,
                    "severity": severity
                })
        else:
            depth_map = None

        # Annotate and write frame
        annotated = annotate_frame(frame, detections, depth_map)
        out.write(annotated)

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to C:\\Projects\\output.mp4")
    print(f"Total frames processed: {frame_count}")

if __name__ == '__main__':
    # Change this to your video file path
    VIDEO_PATH = r"C:\Projects\test_video.mp4"
    run_pipeline(VIDEO_PATH)