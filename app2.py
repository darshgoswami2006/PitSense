import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import math
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

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Projects\PitSense\runs\pothole_v3\weights\best.pt"
OUTPUT_DIR      = r"C:\Projects\PitSense\outputs"
CONFIDENCE      = 0.35
SMOOTH_WINDOW   = 10

# Severity thresholds (for classification only)
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

# Optical-flow speed bands
FLOW_FAST     = 3.5
FLOW_MODERATE = 1.5
SPEED_BAND_COLOR = {
    "FAST":     (0, 0, 220),
    "MODERATE": (0, 140, 255),
    "SLOW":     (0, 200, 80),
}

# Estimated km/h midpoint per speed band
SPEED_BAND_KMH = {
    "FAST":     65,
    "MODERATE": 32,
    "SLOW":     12,
}

# ── Physics-based safe crossing speed ─────────────────────────
def calc_safe_crossing_speed(depth_score, bbox, frame_shape):
    """
    Derives the maximum safe speed to cross a pothole
    from its depth and size — no hardcoded buckets.
    Formula: safe_speed = 40 * e^(-3.5 * depth_score)
    """
    base_safe = 40.0 * math.exp(-3.5 * depth_score)

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area_ratio   = ((x2 - x1) * (y2 - y1)) / (w * h)
    size_penalty = 1.0 - min(area_ratio * 8, 0.35)

    safe_speed = base_safe * size_penalty
    safe_speed = max(5.0, min(40.0, safe_speed))
    return round(safe_speed / 5) * 5

def recommended_speed(detections, speed_band, frame_shape):
    """
    Calculates the speed the car should slow TO based on the
    worst pothole in frame and estimated current vehicle speed.
    Returns (recommended_kmh, urgency) or (None, 'safe').
    """
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

    # Multiple adjacent potholes → extra 15% caution
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
    else:
        return f">  Ease to {rec_speed} km/h"

def advisory_color(rec_speed, urgency):
    if rec_speed is None:
        return (0, 200, 0)
    if urgency == "immediate":
        return (0, 0, 255)
    elif urgency == "soon":
        return (0, 140, 255)
    else:
        return (0, 180, 140)

# ── Severity classifier ───────────────────────────────────────
def classify_severity(depth_score, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    if depth_score > DEPTH_HIGH or area > AREA_HIGH:
        return "HIGH"
    elif depth_score > DEPTH_MEDIUM or area > AREA_MEDIUM:
        return "MEDIUM"
    return "LOW"

# ── Optical-flow speed estimator ─────────────────────────────
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

# ── Drawing helpers ───────────────────────────────────────────
def draw_label_bg(frame, text, origin, font, scale, thickness,
                  bg_color, padding=5):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

# ── Frame annotator ───────────────────────────────────────────
def annotate_frame(frame, detections, frame_count, fps, speed_band):
    h, w    = frame.shape[:2]
    overall = "LOW"
    font    = cv2.FONT_HERSHEY_SIMPLEX

    # Per-detection boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        sev   = det["severity"]
        color = SEV_COLOR[sev]
        thick = 3 if sev == "HIGH" else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        tid   = det.get("track_id", "?")
        safe  = calc_safe_crossing_speed(
            det["depth_score"], det["bbox"], frame.shape)
        label = f"#{tid} [{sev}]  {det['conf']:.2f}  safe:{safe}km/h"
        draw_label_bg(frame, label, (x1, y1 - 6), font, 0.48, 1, color)

        dl = f"Depth: {det['depth_score']:.3f}"
        (dw, dh), _ = cv2.getTextSize(dl, font, 0.45, 1)
        cv2.rectangle(frame, (x1, y2),
                      (x1 + dw + 8, y2 + dh + 8), (30, 30, 30), -1)
        cv2.putText(frame, dl, (x1 + 4, y2 + dh + 2),
                    font, 0.45, color, 1, cv2.LINE_AA)

        if SEVERITY_RANK[sev] > SEVERITY_RANK[overall]:
            overall = sev

    # ── Dynamic speed calculation ─────────────────────────────
    rec_speed, urgency = recommended_speed(detections, speed_band, frame.shape)
    banner_msg = advisory_message(rec_speed, urgency)
    banner_col = advisory_color(rec_speed, urgency)

    # ── Top banner ────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), banner_col, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, banner_msg, (14, 36),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Bottom bar ────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, h - 44), (w, h - 44), (80, 80, 80), 1)

    # Left: pothole count
    cv2.putText(frame, f"Potholes: {len(detections)}", (14, h - 14),
                font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Centre: overall severity
    st = f"Severity: {overall}"
    (sw, _), _ = cv2.getTextSize(st, font, 0.58, 1)
    cv2.putText(frame, st, (w // 2 - sw // 2, h - 14),
                font, 0.58, SEV_COLOR[overall], 1, cv2.LINE_AA)

    # Right: vehicle speed band
    spd_text = f"Vehicle: {speed_band}  |  Frame: {frame_count}"
    (stw, _), _ = cv2.getTextSize(spd_text, font, 0.50, 1)
    cv2.putText(frame, spd_text, (w - stw - 14, h - 14),
                font, 0.50, SPEED_BAND_COLOR[speed_band], 1, cv2.LINE_AA)

    return frame, overall, rec_speed

# ── Main pipeline ─────────────────────────────────────────────
def run_pipeline(video_path, log_fn, progress_fn, done_fn):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        log_fn("Loading YOLO model...")
        yolo = YOLO(YOLO_MODEL_PATH)

        log_fn("Loading depth model...")
        depth_model, depth_transform = load_depth_model()

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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            speed_band  = speed_estimator.update(frame)

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
                        track_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
                    track_history[track_id].append(ds)
                    smoothed_ds = (sum(track_history[track_id])
                                   / len(track_history[track_id]))

                    sev = classify_severity(smoothed_ds, bbox, frame.shape)
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
                frame, detections, frame_count, vid_fps, speed_band)
            out.write(annotated)

            pct = int((frame_count / total) * 100)
            progress_fn(pct)

            if frame_count % 60 == 0:
                spd_str = f"{rec_spd} km/h" if rec_spd else "safe"
                log_fn(f"  Frame {frame_count}/{total} ({pct}%)"
                       f"  |  Vehicle: {speed_band}"
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

# ── GUI ───────────────────────────────────────────────────────
class PitSenseApp:
    def __init__(self, root):
        self.root       = root
        self.video_path = None
        self.root.title("PitSense — Pothole Detection System")
        self.root.geometry("700x700")
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

        # File picker
        drop = tk.Frame(self.root, bg=card, highlightthickness=2,
                        highlightbackground="#2d3250")
        drop.pack(fill="x", padx=20, pady=(16, 0))
        self.file_label = tk.Label(drop, text="No video selected",
                                   font=("Helvetica", 10), bg=card,
                                   fg=sub, anchor="w")
        self.file_label.pack(side="left", padx=14, pady=12,
                             fill="x", expand=True)
        tk.Button(drop, text="Browse Video",
                  font=("Helvetica", 10, "bold"),
                  bg=acc, fg="white", relief="flat",
                  padx=14, pady=6, cursor="hand2",
                  activebackground="#c1121f",
                  activeforeground="white",
                  command=self.browse).pack(
                  side="right", padx=10, pady=8)

        # Progress
        pf = tk.Frame(self.root, bg=bg)
        pf.pack(fill="x", padx=20, pady=(12, 0))
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
        btn_row.pack(pady=(14, 0))

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
        lf.pack(fill="both", expand=True, padx=20, pady=(14, 0))
        tk.Label(lf, text="Processing Log",
                 font=("Helvetica", 9), bg=bg, fg=sub).pack(anchor="w")
        self.log_box = tk.Text(lf, bg=card, fg="#a8dadc",
                               font=("Courier", 9), relief="flat",
                               state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        # Footer
        tk.Label(self.root, text="PitSense  •  Internship Project",
                 font=("Helvetica", 8), bg=bg,
                 fg="#3d405b").pack(pady=(6, 8))

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select a road video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MOV"),
                ("All files", "*.*")])
        if path:
            self.video_path = path
            name = os.path.basename(path)
            size = os.path.getsize(path) / (1024 * 1024)
            self.file_label.config(
                text=f"{name}   ({size:.1f} MB)", fg="#f1faee")
            self.run_btn.config(state="normal")
            self.log(f"Selected: {path}")

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
        if not self.video_path or not DEPS_OK:
            return
        self.run_btn.config(state="disabled", text="Processing...")
        self.progress["value"] = 0

        def worker():
            run_pipeline(
                self.video_path,
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