import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys
from datetime import datetime

# ── Try importing heavy deps, show friendly error if missing ──
try:
    import cv2
    import torch
    from ultralytics import YOLO
    from depth import load_depth_model, estimate_depth, get_depth_score
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    DEPS_ERROR = str(e)

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Projects\PitSense\runs\pothole_v1-5\weights\best.pt"
OUTPUT_DIR      = r"C:\Projects\PitSense\outputs"
CONFIDENCE      = 0.35

DEPTH_HIGH   = 0.3
DEPTH_MEDIUM = 0.15
AREA_HIGH    = 0.04
AREA_MEDIUM  = 0.015

ADVISORY = {
    "HIGH":   {"speed": 10,  "color": (0, 0, 255),   "msg": "!! SLOW DOWN to 10 km/h"},
    "MEDIUM": {"speed": 30,  "color": (0, 165, 255), "msg": ">> Reduce speed to 30 km/h"},
    "LOW":    {"speed": None, "color": (0, 200, 0),  "msg": "OK  Safe to proceed"},
}
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# ── Pipeline logic ────────────────────────────────────────────
def classify_severity(depth_score, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    if depth_score > DEPTH_HIGH or area > AREA_HIGH:
        return "HIGH"
    elif depth_score > DEPTH_MEDIUM or area > AREA_MEDIUM:
        return "MEDIUM"
    return "LOW"

def draw_label_bg(frame, text, origin, font, scale, thickness, bg_color, padding=5):
    import cv2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

def annotate_frame(frame, detections, frame_count, fps):
    import cv2
    h, w = frame.shape[:2]
    overall = "LOW"
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        sev   = det["severity"]
        color = ADVISORY[sev]["color"]
        thick = 3 if sev == "HIGH" else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        draw_label_bg(frame, f"Pothole [{sev}]  {det['conf']:.2f}",
                      (x1, y1 - 6), font, 0.52, 1, color)

        dl = f"Depth: {det['depth_score']:.3f}"
        (dw, dh), _ = cv2.getTextSize(dl, font, 0.45, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + dw + 8, y2 + dh + 8), (30, 30, 30), -1)
        cv2.putText(frame, dl, (x1 + 4, y2 + dh + 2),
                    font, 0.45, color, 1, cv2.LINE_AA)

        if SEVERITY_RANK[sev] > SEVERITY_RANK[overall]:
            overall = sev

    # Top banner
    adv = ADVISORY[overall]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), adv["color"], -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, adv["msg"], (14, 36),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, h - 44), (w, h - 44), (80, 80, 80), 1)

    cv2.putText(frame, f"Potholes: {len(detections)}", (14, h - 14),
                font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    st = f"Severity: {overall}"
    (sw, _), _ = cv2.getTextSize(st, font, 0.62, 1)
    cv2.putText(frame, st, (w // 2 - sw // 2, h - 14),
                font, 0.62, adv["color"], 1, cv2.LINE_AA)

    ts = f"Frame: {frame_count}  |  {fps:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(ts, font, 0.52, 1)
    cv2.putText(frame, ts, (w - tw - 14, h - 14),
                font, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    return frame, overall

def run_pipeline(video_path, log_fn, progress_fn, done_fn):
    import cv2
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

        # Timestamped output filename
        base      = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{base}_pitsense_{timestamp}.mp4"
        out_path  = os.path.join(OUTPUT_DIR, out_name)

        out = cv2.VideoWriter(out_path,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              vid_fps, (vid_w, vid_h))

        log_fn(f"Processing: {os.path.basename(video_path)}")
        log_fn(f"Resolution: {vid_w}x{vid_h}  |  FPS: {vid_fps:.1f}  |  Frames: {total}")
        log_fn(f"Output: {out_name}")
        log_fn("─" * 50)

        frame_count   = 0
        total_potholes = 0
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results    = yolo(frame, conf=CONFIDENCE, verbose=False)[0]
            detections = []

            if results.boxes is not None and len(results.boxes):
                depth_map = estimate_depth(frame, depth_model, depth_transform)
                for box in results.boxes:
                    bbox  = box.xyxy[0].cpu().numpy()
                    conf  = float(box.conf[0])
                    ds    = get_depth_score(depth_map, bbox, frame.shape)
                    sev   = classify_severity(ds, bbox, frame.shape)
                    detections.append({"bbox": bbox, "conf": conf,
                                       "depth_score": ds, "severity": sev})
                    severity_counts[sev] += 1
                total_potholes += len(detections)

            annotated, _ = annotate_frame(frame, detections, frame_count, vid_fps)
            out.write(annotated)

            pct = int((frame_count / total) * 100)
            progress_fn(pct)

            if frame_count % 60 == 0:
                log_fn(f"  Frame {frame_count}/{total} ({pct}%)"
                       f"  —  {len(detections)} pothole(s) this frame")

        cap.release()
        out.release()

        log_fn("─" * 50)
        log_fn(f"Done!  {frame_count} frames processed.")
        log_fn(f"Total pothole detections : {total_potholes}")
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
        self.root  = root
        self.root.title("PitSense — Pothole Detection System")
        self.root.geometry("700x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f1117")

        self.video_path = None
        self._build_ui()

        if not DEPS_OK:
            self.log(f"Missing dependency: {DEPS_ERROR}")
            self.log("Make sure your pothole_env is activated.")

    def _build_ui(self):
        bg   = "#0f1117"
        card = "#1a1d27"
        acc  = "#e63946"
        txt  = "#f1faee"
        sub  = "#8d99ae"

        # ── Header ────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=acc, height=56)
        hdr.pack(fill="x")
        tk.Label(hdr, text="PitSense", font=("Helvetica", 20, "bold"),
                 bg=acc, fg="white").pack(side="left", padx=18, pady=10)
        tk.Label(hdr, text="Pothole Detection & Speed Advisory",
                 font=("Helvetica", 10), bg=acc, fg="white").pack(
                 side="left", pady=10)

        # ── Drop zone ─────────────────────────────────────────
        drop = tk.Frame(self.root, bg=card, bd=0, highlightthickness=2,
                        highlightbackground="#2d3250")
        drop.pack(fill="x", padx=20, pady=(18, 0))

        self.file_label = tk.Label(
            drop, text="No video selected",
            font=("Helvetica", 10), bg=card, fg=sub, anchor="w")
        self.file_label.pack(side="left", padx=14, pady=12, fill="x", expand=True)

        tk.Button(drop, text="Browse Video",
                  font=("Helvetica", 10, "bold"),
                  bg=acc, fg="white", relief="flat",
                  padx=14, pady=6, cursor="hand2",
                  activebackground="#c1121f", activeforeground="white",
                  command=self.browse).pack(side="right", padx=10, pady=8)

        # ── Progress bar ──────────────────────────────────────
        prog_frame = tk.Frame(self.root, bg=bg)
        prog_frame.pack(fill="x", padx=20, pady=(12, 0))

        tk.Label(prog_frame, text="Progress", font=("Helvetica", 9),
                 bg=bg, fg=sub).pack(anchor="w")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Red.Horizontal.TProgressbar",
                        troughcolor=card, background=acc,
                        thickness=14, bordercolor=card)

        self.progress = ttk.Progressbar(prog_frame, style="Red.Horizontal.TProgressbar",
                                        orient="horizontal", length=660, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 0))

        self.pct_label = tk.Label(prog_frame, text="0%",
                                  font=("Helvetica", 9), bg=bg, fg=sub)
        self.pct_label.pack(anchor="e")

        # ── Run button ────────────────────────────────────────
        self.run_btn = tk.Button(
            self.root, text="Run PitSense",
            font=("Helvetica", 12, "bold"),
            bg=acc, fg="white", relief="flat",
            padx=20, pady=10, cursor="hand2",
            activebackground="#c1121f", activeforeground="white",
            state="disabled", command=self.start_processing)
        self.run_btn.pack(pady=(14, 0))

        # ── Log area ──────────────────────────────────────────
        log_frame = tk.Frame(self.root, bg=bg)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(14, 0))

        tk.Label(log_frame, text="Processing Log",
                 font=("Helvetica", 9), bg=bg, fg=sub).pack(anchor="w")

        self.log_box = tk.Text(
            log_frame, bg=card, fg="#a8dadc",
            font=("Courier", 9), relief="flat",
            state="disabled", wrap="word",
            insertbackground="white")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        scrollbar = tk.Scrollbar(self.log_box)
        self.log_box.configure(yscrollcommand=scrollbar.set)

        # ── Output button (hidden until done) ─────────────────
        self.open_btn = tk.Button(
            self.root, text="Open Output Folder",
            font=("Helvetica", 10), bg="#457b9d", fg="white",
            relief="flat", padx=14, pady=7, cursor="hand2",
            command=self.open_output_folder)

        # ── Footer ────────────────────────────────────────────
        tk.Label(self.root, text="PitSense  •  Internship Project",
                 font=("Helvetica", 8), bg=bg, fg="#3d405b").pack(pady=(6, 8))

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select a road video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MOV"),
                       ("All files", "*.*")])
        if path:
            self.video_path = path
            name = os.path.basename(path)
            size = os.path.getsize(path) / (1024 * 1024)
            self.file_label.config(
                text=f"{name}   ({size:.1f} MB)", fg="#f1faee")
            self.run_btn.config(state="normal")
            self.open_btn.pack_forget()
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
        self.open_btn.pack_forget()

        def worker():
            run_pipeline(
                self.video_path,
                log_fn      = lambda m: self.root.after(0, self.log, m),
                progress_fn = lambda p: self.root.after(0, self.set_progress, p),
                done_fn     = lambda p: self.root.after(0, self.on_done, p),
            )

        threading.Thread(target=worker, daemon=True).start()

    def on_done(self, out_path):
        self.run_btn.config(state="normal", text="Run PitSense")
        if out_path:
            self.set_progress(100)
            self.open_btn.pack(pady=(8, 0))
            self.log(f"\nOutput ready — click 'Open Output Folder' to view.")
        else:
            self.log("Processing failed. Check the log above.")

    def open_output_folder(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.startfile(OUTPUT_DIR)

if __name__ == "__main__":
    root = tk.Tk()
    app  = PitSenseApp(root)
    root.mainloop()