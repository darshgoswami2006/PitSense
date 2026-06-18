import cv2
import numpy as np
import torch
from ultralytics import YOLO
from depth import load_depth_model, estimate_depth, get_depth_score

# ── Configuration ────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Projects\PitSense\runs\pothole_v1-5\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.35

# Severity thresholds
DEPTH_HIGH   = 0.3
DEPTH_MEDIUM = 0.15
AREA_HIGH    = 0.04
AREA_MEDIUM  = 0.015

# Speed advisory
ADVISORY = {
    "HIGH":   {"slow": True,  "speed": 10,  "color": (0, 0, 255),   "msg": "!! SLOW DOWN to 10 km/h"},
    "MEDIUM": {"slow": True,  "speed": 30,  "color": (0, 165, 255), "msg": ">> Reduce speed to 30 km/h"},
    "LOW":    {"slow": False, "speed": None, "color": (0, 200, 0),  "msg": "OK  Safe to proceed"},
}

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

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

# ── Draw rounded rectangle (for cleaner labels) ──────────────
def draw_label_bg(frame, text, origin, font, scale, thickness, bg_color, padding=5):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ── Annotate Frame ────────────────────────────────────────────
def annotate_frame(frame, detections, frame_count, fps):
    h, w = frame.shape[:2]
    overall_severity = "LOW"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Per-detection annotations ─────────────────────────────
    for det in detections:
        bbox       = det["bbox"]
        conf       = det["conf"]
        severity   = det["severity"]
        depth_score = det["depth_score"]
        color      = ADVISORY[severity]["color"]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Bounding box — thicker for HIGH severity
        thickness = 3 if severity == "HIGH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Top label: severity + confidence
        top_label = f"Pothole [{severity}]  {conf:.2f}"
        draw_label_bg(frame, top_label, (x1, y1 - 6), font, 0.52, 1, color)

        # Bottom label: depth score
        depth_label = f"Depth score: {depth_score:.3f}"
        (dw, dh), _ = cv2.getTextSize(depth_label, font, 0.45, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + dw + 8, y2 + dh + 8), (30, 30, 30), -1)
        cv2.putText(frame, depth_label, (x1 + 4, y2 + dh + 2),
                    font, 0.45, color, 1, cv2.LINE_AA)

        # Track worst severity
        if SEVERITY_RANK[severity] > SEVERITY_RANK[overall_severity]:
            overall_severity = severity

    # ── Top banner (speed advisory) ───────────────────────────
    adv          = ADVISORY[overall_severity]
    banner_color = adv["color"]
    banner_h     = 58

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Advisory text — left side
    cv2.putText(frame, adv["msg"], (14, 36),
            font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Bottom bar ────────────────────────────────────────────
    bar_h = 44
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    # Divider line
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), (80, 80, 80), 1)

    # Left: pothole count
    count_text = f"Potholes: {len(detections)}"
    cv2.putText(frame, count_text, (14, h - 14),
                font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    # Centre: overall severity badge
    sev_text  = f"Severity: {overall_severity}"
    sev_color = ADVISORY[overall_severity]["color"]
    (sew, _), _ = cv2.getTextSize(sev_text, font, 0.62, 1)
    cv2.putText(frame, sev_text, (w // 2 - sew // 2, h - 14),
                font, 0.62, sev_color, 1, cv2.LINE_AA)

    # Right: frame / timestamp
    timestamp = f"Frame: {frame_count}  |  {fps:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(timestamp, font, 0.52, 1)
    cv2.putText(frame, timestamp, (w - tw - 14, h - 14),
                font, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

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

    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = cv2.VideoWriter(
        r"C:\Projects\PitSense\output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        vid_fps, (vid_w, vid_h)
    )

    frame_count = 0
    print("Processing video...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results    = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []

        if results.boxes is not None and len(results.boxes):
            depth_map = estimate_depth(frame, depth_model, depth_transform)

            for box in results.boxes:
                bbox        = box.xyxy[0].cpu().numpy()
                conf        = float(box.conf[0])
                depth_score = get_depth_score(depth_map, bbox, frame.shape)
                severity    = classify_severity(depth_score, bbox, frame.shape)

                detections.append({
                    "bbox":        bbox,
                    "conf":        conf,
                    "depth_score": depth_score,
                    "severity":    severity,
                })
        else:
            depth_map = None

        annotated = annotate_frame(frame, detections, frame_count, vid_fps)
        out.write(annotated)

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to C:\\Projects\\PitSense\\output.mp4")
    print(f"Total frames processed: {frame_count}")

if __name__ == '__main__':
    VIDEO_PATH = r"C:\Projects\PitSense\test_video.mp4"
    run_pipeline(VIDEO_PATH)