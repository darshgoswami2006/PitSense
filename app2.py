import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import math
import re
from datetime import datetime
from collections import deque

try:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from depth import load_depth_model, estimate_depth, get_depth_score
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    DEPS_ERROR = str(e)

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

YOLO_MODEL_PATH = r"C:\Projects\PitSense\runs\pothole_v3\weights\best.pt"
OUTPUT_DIR      = r"C:\Projects\PitSense\outputs"
CONFIDENCE      = 0.35
SMOOTH_WINDOW   = 10

DEPTH_HIGH   = 0.3
DEPTH_MEDIUM = 0.15
AREA_HIGH    = 0.04
AREA_MEDIUM  = 0.015

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
SEV_COLOR = {
    "HIGH":   (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW":    (0, 200, 0),
}

FLOW_FAST     = 15.0
FLOW_MODERATE = 3.0

SPEED_BAND_COLOR = {
    "FAST":     (0, 0, 220),
    "MODERATE": (0, 140, 255),
    "SLOW":     (0, 200, 80),
}

SPEED_BAND_KMH = {
    "FAST":     90,
    "MODERATE": 45,
    "SLOW":     15,
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

class OCRSpeedReader:
    """
    Reads GPS speed burned into dashcam footage.
    Supports formats: 096km/h, 65mph, 72kph, 096kmh
    Initialises EasyOCR once and caches the last known speed.
    """
    def __init__(self):
        self.reader     = None
        self.last_speed = None   
        self.ready      = False

        if OCR_AVAILABLE:
            try:
                gpu = torch.cuda.is_available() if 'torch' in dir() else False
                self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
                self.ready  = True
            except Exception:
                self.ready = False

    def read(self, frame):
        """
        Tries to extract speed from the bottom portion of the frame.
        Returns km/h as int or None.
        """
        if not self.ready or self.reader is None:
            return None
        try:
            h, w  = frame.shape[:2]
            # Crop bottom 35% of frame 
            roi   = frame[int(h * 0.65):, :]
            texts = self.reader.readtext(roi, detail=0)
            text  = " ".join(texts).lower().replace(" ", "")

            # Match km/h variants
            m = re.search(r'(\d{1,3})[o0]?\s*k[mp]h?', text)
            if m:
                spd = int(m.group(1))
                if 0 < spd < 300:
                    self.last_speed = spd
                    return spd

            # Match mph and convert
            m = re.search(r'(\d{1,3})\s*mph', text)
            if m:
                spd = int(int(m.group(1)) * 1.609)
                if 0 < spd < 300:
                    self.last_speed = spd
                    return spd

        except Exception:
            pass
        return None

    def kmh_to_band(self, kmh):
        if kmh is None:
            return None
        if kmh >= 80:
            return "FAST"
        elif kmh >= 20:
            return "MODERATE"
        return "SLOW"

def calc_safe_crossing_speed(depth_score, bbox, frame_shape):
    base_safe    = 40.0 * math.exp(-3.5 * depth_score)
    h, w         = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area_ratio   = ((x2 - x1) * (y2 - y1)) / (w * h)
    size_penalty = 1.0 - min(area_ratio * 8, 0.35)
    safe_speed   = max(5.0, min(40.0, base_safe * size_penalty))
    return round(safe_speed / 5) * 5

def recommended_speed(detections, speed_band, frame_shape):
    if not detections:
        return None, "safe"
    worst_safe = float("inf")
    for det in detections:
        safe = calc_safe_crossing_speed(
            det["depth_score"], det["bbox"], frame_shape)
        if safe < worst_safe:
            worst_safe = safe
    current_est = SPEED_BAND_KMH[speed_band]
    if current_est <= worst_safe:
        return None, "safe"
    if len(detections) > 1:
        worst_safe = max(5, worst_safe * 0.85)
        worst_safe = round(worst_safe / 5) * 5
    reduction = current_est - worst_safe
    if reduction > 35:
        urgency = "immediate"
    elif reduction > 15:
        urgency = "soon"
    else:
        urgency = "gentle"
    return int(worst_safe), urgency

def advisory_message(rec_speed, urgency):
    if rec_speed is None:
        return "OK  Safe to proceed"
    if urgency == "immediate":
        return f"!! BRAKE NOW — Reduce to {rec_speed} km/h"
    elif urgency == "soon":
        return f">> Reduce speed to {rec_speed} km/h"
    return f">  Ease to {rec_speed} km/h"

def advisory_color(rec_speed, urgency):
    if rec_speed is None:
        return (0, 200, 0)
    if urgency == "immediate":
        return (0, 0, 255)
    elif urgency == "soon":
        return (0, 140, 255)
    return (0, 180, 140)

def classify_severity(depth_score, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    if depth_score > DEPTH_HIGH or area > AREA_HIGH:
        return "HIGH"
    elif depth_score > DEPTH_MEDIUM or area > AREA_MEDIUM:
        return "MEDIUM"
    return "LOW"

class SpeedEstimator:
    def __init__(self, history=15):
        self.prev_gray   = None
        self.mag_history = deque(maxlen=history)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h    = gray.shape[0]
        roi  = gray[int(h * 0.45):]
        if self.prev_gray is not None:
            prev_roi = self.prev_gray[int(h * 0.45):]
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            self.mag_history.append(mag)
        else:
            self.mag_history.append(0.0)
        self.prev_gray = gray
        avg = sum(self.mag_history) / len(self.mag_history)
        if avg >= FLOW_FAST:
            return "FAST"
        elif avg >= FLOW_MODERATE:
            return "MODERATE"
        return "SLOW"

def draw_label_bg(frame, text, origin, font, scale,
                  thickness, bg_color, padding=5):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

def annotate_frame(frame, detections, speed_band,
                   speed_source="flow", actual_kmh=None,
                   frame_count=None, fps=None, is_image=False):
    h, w    = frame.shape[:2]
    overall = "LOW"
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        sev   = det["severity"]
        color = SEV_COLOR[sev]
        thick = 3 if sev == "HIGH" else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        safe  = calc_safe_crossing_speed(
            det["depth_score"], det["bbox"], frame.shape)
        tid   = det.get("track_id", "")
        tid_s = f"#{tid} " if tid != "" else ""
        label = f"{tid_s}[{sev}]  {det['conf']:.2f}  safe:{safe}km/h"
        draw_label_bg(frame, label, (x1, y1 - 6), font, 0.48, 1, color)

        dl = f"Depth: {det['depth_score']:.3f}"
        (dw, dh), _ = cv2.getTextSize(dl, font, 0.45, 1)
        cv2.rectangle(frame, (x1, y2),
                      (x1 + dw + 8, y2 + dh + 8), (30, 30, 30), -1)
        cv2.putText(frame, dl, (x1 + 4, y2 + dh + 2),
                    font, 0.45, color, 1, cv2.LINE_AA)

        if SEVERITY_RANK[sev] > SEVERITY_RANK[overall]:
            overall = sev

    rec_speed, urgency = recommended_speed(
        detections, speed_band, frame.shape)
    banner_msg = advisory_message(rec_speed, urgency)
    banner_col = advisory_color(rec_speed, urgency)

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), banner_col, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, banner_msg, (14, 36),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, h - 44), (w, h - 44), (80, 80, 80), 1)

    # Left: pothole count
    cv2.putText(frame, f"Potholes: {len(detections)}", (14, h - 14),
                font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Centre: severity
    st = f"Severity: {overall}"
    (sw, _), _ = cv2.getTextSize(st, font, 0.58, 1)
    cv2.putText(frame, st, (w // 2 - sw // 2, h - 14),
                font, 0.58, SEV_COLOR[overall], 1, cv2.LINE_AA)

    # Right: vehicle speed 
    if is_image:
        right_text = "IMAGE MODE"
        right_col  = (180, 180, 180)
    elif actual_kmh is not None:
        right_text = f"GPS: {actual_kmh} km/h ({speed_band})  |  F:{frame_count}"
        right_col  = SPEED_BAND_COLOR[speed_band]
    else:
        right_text = f"Vehicle: {speed_band} [flow]  |  F:{frame_count}"
        right_col  = SPEED_BAND_COLOR[speed_band]

    (rtw, _), _ = cv2.getTextSize(right_text, font, 0.48, 1)
    cv2.putText(frame, right_text, (w - rtw - 14, h - 14),
                font, 0.48, right_col, 1, cv2.LINE_AA)

    return frame, overall, rec_speed

def process_image(img_path, log_fn, progress_fn, done_fn):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        progress_fn(10)

        log_fn("Loading YOLO model...")
        yolo = YOLO(YOLO_MODEL_PATH)
        progress_fn(30)

        log_fn("Loading depth model...")
        depth_model, depth_transform = load_depth_model()
        progress_fn(50)

        frame = cv2.imread(img_path)
        if frame is None:
            log_fn(f"ERROR: Cannot read image: {img_path}")
            done_fn(None)
            return

        log_fn(f"Processing : {os.path.basename(img_path)}")
        log_fn(f"Resolution : {frame.shape[1]}x{frame.shape[0]}")
        progress_fn(60)

        results    = yolo(frame, conf=CONFIDENCE, verbose=False)[0]
        detections = []

        if results.boxes is not None and len(results.boxes):
            depth_map = estimate_depth(frame, depth_model, depth_transform)
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                ds   = get_depth_score(depth_map, bbox, frame.shape)
                sev  = classify_severity(ds, bbox, frame.shape)
                detections.append({
                    "bbox": bbox, "conf": conf,
                    "depth_score": ds, "severity": sev,
                })

        progress_fn(80)

        annotated, overall, rec_spd = annotate_frame(
            frame, detections, "MODERATE", is_image=True)

        base      = os.path.splitext(os.path.basename(img_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{base}_pitsense_{timestamp}.png"
        out_path  = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        progress_fn(100)

        log_fn("─" * 52)
        log_fn(f"Potholes detected : {len(detections)}")
        for det in detections:
            safe = calc_safe_crossing_speed(
                det["depth_score"], det["bbox"], frame.shape)
            log_fn(f"  [{det['severity']}]  conf:{det['conf']:.2f}"
                   f"  depth:{det['depth_score']:.3f}"
                   f"  safe:{safe} km/h")
        spd_str = f"{rec_spd} km/h" if rec_spd else "safe to proceed"
        log_fn(f"Advisory          : {spd_str}")
        log_fn(f"Saved to          : outputs/{out_name}")
        done_fn(out_path)

    except Exception as e:
        log_fn(f"ERROR: {e}")
        done_fn(None)

def process_video(video_path, log_fn, progress_fn, done_fn):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        log_fn("Loading YOLO model...")
        yolo = YOLO(YOLO_MODEL_PATH)

        log_fn("Loading depth model...")
        depth_model, depth_transform = load_depth_model()

        # Initialise OCR speed reader
        ocr = OCRSpeedReader()
        if ocr.ready:
            log_fn("OCR speed reader : enabled (will read dashcam GPS overlay)")
        else:
            log_fn("OCR speed reader : not available — using optical flow")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_fn(f"ERROR: Cannot open video: {video_path}")
            done_fn(None)
            return

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        base      = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{base}_pitsense_{timestamp}.mp4"
        out_path  = os.path.join(OUTPUT_DIR, out_name)

        out = cv2.VideoWriter(out_path,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              vid_fps, (vid_w, vid_h))

        log_fn(f"Processing : {os.path.basename(video_path)}")
        log_fn(f"Resolution : {vid_w}x{vid_h}  |  "
               f"FPS: {vid_fps:.1f}  |  Frames: {total}")
        log_fn(f"Output     : {out_name}")
        log_fn("─" * 52)

        frame_count     = 0
        total_potholes  = 0
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        speed_estimator = SpeedEstimator(history=15)
        track_history   = {}

        cached_kmh      = None
        OCR_INTERVAL    = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            
            flow_band = speed_estimator.update(frame)

            
            actual_kmh  = cached_kmh
            speed_source = "flow"

            if ocr.ready and frame_count % OCR_INTERVAL == 0:
                result = ocr.read(frame)
                if result is not None:
                    cached_kmh = result
                    actual_kmh = result

            if actual_kmh is not None:
                speed_band   = ocr.kmh_to_band(actual_kmh)
                speed_source = "ocr"
            else:
                speed_band = flow_band

           
            results    = yolo.track(frame, conf=CONFIDENCE, persist=True,
                                    tracker="bytetrack.yaml",
                                    verbose=False)[0]
            detections = []

            if results.boxes is not None and len(results.boxes):
                depth_map = estimate_depth(
                    frame, depth_model, depth_transform)

                for box in results.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id[0])
                    bbox     = box.xyxy[0].cpu().numpy()
                    conf     = float(box.conf[0])
                    ds       = get_depth_score(
                        depth_map, bbox, frame.shape)

                    if track_id not in track_history:
                        track_history[track_id] = deque(
                            maxlen=SMOOTH_WINDOW)
                    track_history[track_id].append(ds)
                    smoothed_ds = (sum(track_history[track_id])
                                   / len(track_history[track_id]))

                    sev = classify_severity(
                        smoothed_ds, bbox, frame.shape)
                    detections.append({
                        "bbox":        bbox,
                        "conf":        conf,
                        "depth_score": smoothed_ds,
                        "severity":    sev,
                        "track_id":    track_id,
                    })
                    severity_counts[sev] += 1
                total_potholes += len(detections)

            annotated, _, rec_spd = annotate_frame(
                frame, detections, speed_band,
                speed_source=speed_source,
                actual_kmh=actual_kmh,
                frame_count=frame_count,
                fps=vid_fps,
                is_image=False)
            out.write(annotated)

            pct = int((frame_count / total) * 100)
            progress_fn(pct)

            if frame_count % 60 == 0:
                spd_str  = f"{rec_spd} km/h" if rec_spd else "safe"
                src_str  = f"GPS:{actual_kmh}km/h" if actual_kmh else f"flow:{speed_band}"
                log_fn(f"  Frame {frame_count}/{total} ({pct}%)"
                       f"  |  Speed: {src_str}"
                       f"  |  Advisory: {spd_str}"
                       f"  |  {len(detections)} pothole(s)")

        cap.release()
        out.release()

        log_fn("─" * 52)
        log_fn(f"Done!  {frame_count} frames processed.")
        log_fn(f"Total detections : {total_potholes}")
        log_fn(f"  LOW    : {severity_counts['LOW']}")
        log_fn(f"  MEDIUM : {severity_counts['MEDIUM']}")
        log_fn(f"  HIGH   : {severity_counts['HIGH']}")
        log_fn(f"Saved to : outputs/{out_name}")
        done_fn(out_path)

    except Exception as e:
        log_fn(f"ERROR: {e}")
        done_fn(None)


def run_pipeline(file_path, log_fn, progress_fn, done_fn):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        process_image(file_path, log_fn, progress_fn, done_fn)
    elif ext in VIDEO_EXTS:
        process_video(file_path, log_fn, progress_fn, done_fn)
    else:
        log_fn(f"ERROR: Unsupported file type '{ext}'")
        done_fn(None)

class PitSenseApp:
    def __init__(self, root):
        self.root      = root
        self.file_path = None
        self.root.title("PitSense — Pothole Detection System")
        self.root.geometry("700x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f1117")
        self._build_ui()

        if not DEPS_OK:
            self.log(f"Missing dependency: {DEPS_ERROR}")
            self.log("Make sure your pothole_env is activated.")

    def _build_ui(self):
        bg   = "#0f1117"
        card = "#1a1d27"
        acc  = "#e63946"
        sub  = "#8d99ae"

        # Header
        hdr = tk.Frame(self.root, bg=acc, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="PitSense",
                 font=("Helvetica", 20, "bold"),
                 bg=acc, fg="white").pack(side="left", padx=18, pady=10)
        tk.Label(hdr, text="Pothole Detection & Speed Advisory",
                 font=("Helvetica", 10),
                 bg=acc, fg="#ffcdd2").pack(side="left", pady=10)

        # OCR status badge (top right of header)
        ocr_color = "#2d6a4f" if OCR_AVAILABLE else "#555"
        ocr_text  = "OCR ON" if OCR_AVAILABLE else "OCR OFF"
        tk.Label(hdr, text=ocr_text,
                 font=("Helvetica", 8, "bold"),
                 bg=ocr_color, fg="white",
                 padx=6, pady=3).pack(side="right", padx=12)

        # File picker
        drop = tk.Frame(self.root, bg=card, highlightthickness=2,
                        highlightbackground="#2d3250")
        drop.pack(fill="x", padx=20, pady=(16, 0))
        self.file_label = tk.Label(
            drop, text="No file selected",
            font=("Helvetica", 10), bg=card, fg=sub, anchor="w")
        self.file_label.pack(side="left", padx=14, pady=12,
                             fill="x", expand=True)
        tk.Button(drop, text="Browse Image",
                  font=("Helvetica", 9, "bold"),
                  bg="#2d6a4f", fg="white", relief="flat",
                  padx=10, pady=6, cursor="hand2",
                  activebackground="#1b4332",
                  activeforeground="white",
                  command=self.browse_image).pack(
                  side="right", padx=(4, 10), pady=8)
        tk.Button(drop, text="Browse Video",
                  font=("Helvetica", 9, "bold"),
                  bg=acc, fg="white", relief="flat",
                  padx=10, pady=6, cursor="hand2",
                  activebackground="#c1121f",
                  activeforeground="white",
                  command=self.browse_video).pack(
                  side="right", padx=0, pady=8)

        # Mode + OCR info line
        self.type_label = tk.Label(
            self.root, text="",
            font=("Helvetica", 9), bg=bg, fg=sub)
        self.type_label.pack(anchor="e", padx=22)

        # Progress
        pf = tk.Frame(self.root, bg=bg)
        pf.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(pf, text="Progress", font=("Helvetica", 9),
                 bg=bg, fg=sub).pack(anchor="w")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Red.Horizontal.TProgressbar",
                        troughcolor=card, background=acc,
                        thickness=14, bordercolor=card)
        self.progress = ttk.Progressbar(
            pf, style="Red.Horizontal.TProgressbar",
            orient="horizontal", length=660, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 0))
        self.pct_label = tk.Label(pf, text="0%",
                                  font=("Helvetica", 9), bg=bg, fg=sub)
        self.pct_label.pack(anchor="e")

        # Buttons row
        btn_row = tk.Frame(self.root, bg=bg)
        btn_row.pack(pady=(12, 0))

        self.run_btn = tk.Button(
            btn_row, text="Run PitSense",
            font=("Helvetica", 12, "bold"),
            bg=acc, fg="white", relief="flat",
            padx=20, pady=10, cursor="hand2",
            activebackground="#c1121f", activeforeground="white",
            state="disabled", command=self.start_processing)
        self.run_btn.pack(side="left", padx=(0, 14))

        self.open_btn = tk.Button(
            btn_row, text="Open Output Folder",
            font=("Helvetica", 11),
            bg="#457b9d", fg="white", relief="flat",
            padx=16, pady=10, cursor="hand2",
            activebackground="#1d6c8a", activeforeground="white",
            command=self.open_output_folder)
        self.open_btn.pack(side="left")

        # Processing log
        lf = tk.Frame(self.root, bg=bg)
        lf.pack(fill="both", expand=True, padx=20, pady=(12, 0))
        tk.Label(lf, text="Processing Log",
                 font=("Helvetica", 9), bg=bg, fg=sub).pack(anchor="w")
        self.log_box = tk.Text(
            lf, bg=card, fg="#a8dadc",
            font=("Courier", 9), relief="flat",
            state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        # Footer
        ocr_note = "OCR speed reading enabled" if OCR_AVAILABLE \
                   else "Install easyocr for GPS speed reading"
        tk.Label(self.root,
                 text=f"PitSense  •  Internship Project  •  {ocr_note}",
                 font=("Helvetica", 8), bg=bg,
                 fg="#3d405b").pack(pady=(6, 8))

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select a road video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")])
        if path:
            self._set_file(path, "VIDEO")

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select a road image",
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")])
        if path:
            self._set_file(path, "IMAGE")

    def _set_file(self, path, kind):
        self.file_path = path
        name = os.path.basename(path)
        size = os.path.getsize(path) / (1024 * 1024)
        self.file_label.config(
            text=f"{name}   ({size:.1f} MB)", fg="#f1faee")

        if kind == "IMAGE":
            self.type_label.config(
                text="Mode: IMAGE  — single frame, speed defaults to MODERATE",
                fg="#a8dadc")
        else:
            ocr_str = "OCR + optical flow" if OCR_AVAILABLE \
                      else "optical flow only"
            self.type_label.config(
                text=f"Mode: VIDEO  — speed detection: {ocr_str}",
                fg="#a8dadc")

        self.run_btn.config(state="normal")
        self.log(f"Selected [{kind}]: {path}")

    def log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def set_progress(self, pct):
        self.progress["value"] = pct
        self.pct_label.config(text=f"{pct}%")
        self.root.update_idletasks()

    def start_processing(self):
        if not self.file_path or not DEPS_OK:
            return
        self.run_btn.config(state="disabled", text="Processing...")
        self.progress["value"] = 0

        def worker():
            run_pipeline(
                self.file_path,
                log_fn      = lambda m: self.root.after(0, self.log, m),
                progress_fn = lambda p: self.root.after(
                    0, self.set_progress, p),
                done_fn     = lambda p: self.root.after(
                    0, self.on_done, p),
            )

        threading.Thread(target=worker, daemon=True).start()

    def on_done(self, out_path):
        self.run_btn.config(state="normal", text="Run PitSense")
        if out_path:
            self.set_progress(100)
            self.log("\nOutput ready — click 'Open Output Folder' to view.")
        else:
            self.log("Processing failed. Check the log above.")

    def open_output_folder(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.startfile(OUTPUT_DIR)

if __name__ == "__main__":
    root = tk.Tk()
    app  = PitSenseApp(root)
    root.mainloop()