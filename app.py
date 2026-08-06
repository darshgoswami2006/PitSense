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
    from depth import load_depth_model, estimate_depth, get_depth_score  # MiDaS depth
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    DEPS_ERROR = str(e)

# EasyOCR is optional
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configuration constants
YOLO_MODEL_PATH = r"C:\Projects\PitSense\runs\pothole_v3\weights\best.pt"
OUTPUT_DIR      = r"C:\Projects\PitSense\outputs"  # where output files are saved
CONFIDENCE      = 0.35       # minimum YOLO confidence to accept a detection
SMOOTH_WINDOW   = 10         # number of frames to average depth scores over per track

# Depth score thresholds for severity classification
# These compare against the relative depth score d (0 to 1)
DEPTH_HIGH   = 0.3           # above this -> HIGH severity
DEPTH_MEDIUM = 0.15          # above this -> MEDIUM severity

# Bounding box area ratio thresholds (pothole area / frame area)
AREA_HIGH    = 0.04          # pothole covers >4% of frame -> HIGH
AREA_MEDIUM  = 0.015         # pothole covers >1.5% of frame -> MEDIUM

# Numeric rank for severity 
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# BGR colour for each severity level (used for bounding boxes and labels)
SEV_COLOR = {
    "HIGH":   (0, 0, 255),    # red
    "MEDIUM": (0, 165, 255),  # orange
    "LOW":    (0, 200, 0),    # green
}

# Optical flow thresholds (may need to change based on quality of video)
FLOW_FAST     = 15.0   # above this -> FAST
FLOW_MODERATE = 3.0    # above this -> MODERATE, below -> SLOW

# BGR colour for each speed band shown in the bottom bar
SPEED_BAND_COLOR = {
    "FAST":     (0, 0, 220),
    "MODERATE": (0, 140, 255),
    "SLOW":     (0, 200, 80),
}

# Estimated real-world speed (km/h) for each band
# These are midpoint estimates since optical flow gives relative, not absolute speed
SPEED_BAND_KMH = {
    "FAST":     80,
    "MODERATE": 50,
    "SLOW":     20,
}

# Supported file extensions for the file picker
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


#OCR Speed Reader 
class OCRSpeedReader:
    """
    Reads GPS speed burned into dashcam footage.
    Supports formats: 096km/h, 65mph, 72kph, 096kmh
    Initialises EasyOCR once and caches the last known speed.
    """
    def __init__(self):
        self.reader     = None
        self.last_speed = None    # stores the last successfully read speed (km/h)
        self.ready      = False   

        if OCR_AVAILABLE:
            try:
                # Use GPU if available for faster OCR, otherwise fall back to CPU
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

            # Crop to bottom 35% of the frame
            roi   = frame[int(h * 0.65):, :]

            # Run OCR on the cropped region, return text only (no bounding boxes)
            texts = self.reader.readtext(roi, detail=0)

            # Join all detected text into one string and clean it up
            text  = " ".join(texts).lower().replace(" ", "")

            # Match km/h variants using regex (handles OCR errors like 'o' instead of '0')
            m = re.search(r'(\d{1,3})[o0]?\s*k[mp]h?', text)
            if m:
                spd = int(m.group(1))
                # Sanity check: reject anything outside realistic road speed range
                if 0 < spd < 300:
                    self.last_speed = spd
                    return spd

            # Match mph and convert to km/h (1 mph = 1.609 km/h)
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
        # Convert a real km/h value from OCR into a speed band label
        if kmh is None:
            return None
        if kmh >= 80:
            return "FAST"
        elif kmh >= 20:
            return "MODERATE"
        return "SLOW"


#Physics-based safe crossing speed
def calc_safe_crossing_speed(depth_score, bbox, frame_shape):
    # Exponential decay formula: deeper pothole = exponentially lower safe speed
    # 40 km/h is the max safe speed (for a perfectly flat road, d=0)
    # 3.5 controls how steeply speed drops as depth increases
    base_safe    = 40.0 * math.exp(-3.5 * depth_score)

    h, w         = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate pothole area as a fraction of total frame area
    area_ratio   = ((x2 - x1) * (y2 - y1)) / (w * h)

    # Size penalty: larger pothole → additional speed reduction (max 35%)
    size_penalty = 1.0 - min(area_ratio * 8, 0.35)

    # Apply size penalty, clamp between 5 km/h (minimum) and 40 km/h (maximum)
    safe_speed   = max(5.0, min(40.0, base_safe * size_penalty))

    # Round to nearest 5 km/h to match real road sign conventions
    return round(safe_speed / 5) * 5


def recommended_speed(detections, speed_band, frame_shape):
    # If no potholes in this frame, no advisory needed
    if not detections:
        return None, "safe"

    # Find the worst-case (lowest) safe speed across all detected potholes
    worst_safe = float("inf")
    for det in detections:
        safe = calc_safe_crossing_speed(
            det["depth_score"], det["bbox"], frame_shape)
        if safe < worst_safe:
            worst_safe = safe

    # Get the estimated current vehicle speed from the speed band
    current_est = SPEED_BAND_KMH[speed_band]

    # If the car is already at or below the safe speed, no action needed
    if current_est <= worst_safe:
        return None, "safe"

    # Multiple adjacent potholes are more dangerous
    if len(detections) > 1:
        worst_safe = max(5, worst_safe * 0.85)
        worst_safe = round(worst_safe / 5) * 5

    # Determine urgency based on how much speed reduction is needed
    reduction = current_est - worst_safe
    if reduction > 35:
        urgency = "immediate"  # large speed gap, brake hard
    elif reduction > 15:
        urgency = "soon"       # moderate gap, reduce speed
    else:
        urgency = "gentle"     # small gap, ease off

    return int(worst_safe), urgency


def advisory_message(rec_speed, urgency):
    # Build the banner text shown at the top of the frame
    if rec_speed is None:
        return "OK  Safe to proceed"
    if urgency == "immediate":
        return f"!! BRAKE NOW — Reduce to {rec_speed} km/h"
    elif urgency == "soon":
        return f">> Reduce speed to {rec_speed} km/h"
    return f">  Ease to {rec_speed} km/h"


def advisory_color(rec_speed, urgency):
    # Choose banner colour based on urgency level (BGR format)
    if rec_speed is None:
        return (0, 200, 0)       # green, safe
    if urgency == "immediate":
        return (0, 0, 255)       # red, brake now
    elif urgency == "soon":
        return (0, 140, 255)     # orange, reduce speed
    return (0, 180, 140)         


#Severity Classifier
def classify_severity(depth_score, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate pothole area as fraction of frame
    area = ((x2 - x1) * (y2 - y1)) / (w * h)

    # Either deep OR large enough -> HIGH
    if depth_score > DEPTH_HIGH or area > AREA_HIGH:
        return "HIGH"
    # Either moderately deep OR moderately large -> MEDIUM
    elif depth_score > DEPTH_MEDIUM or area > AREA_MEDIUM:
        return "MEDIUM"
    return "LOW"


#Optical Flow Speed Estimator
class SpeedEstimator:
    def __init__(self, history=15):
        self.prev_gray   = None
        # Rolling history of flow magnitudes
        self.mag_history = deque(maxlen=history)

    def update(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h    = gray.shape[0]

        # Only use the bottom 55% of the frame (road region)
        # Avoids sky and distant scenery which create unreliable flow vectors
        roi  = gray[int(h * 0.45):]

        if self.prev_gray is not None:
            prev_roi = self.prev_gray[int(h * 0.45):]

            # Farneback dense optical flow: computes a motion vector for every pixel
            # Returns an array of shape (H, W, 2) where [:,:,0]=horizontal, [:,:,1]=vertical
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # Euclidean magnitude of each flow vector: sqrt(fx^2 + fy^2)
            # Then take the mean across all road pixels
            mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            self.mag_history.append(mag)
        else:
            # First frame
            self.mag_history.append(0.0)

        # Store current frame for comparison in the next call
        self.prev_gray = gray

        # Average over the last 15 frames to smooth out per-frame noise
        avg = sum(self.mag_history) / len(self.mag_history)

        # Classify into speed band using thresholds
        if avg >= FLOW_FAST:
            return "FAST"
        elif avg >= FLOW_MODERATE:
            return "MODERATE"
        return "SLOW"


# Drawing Helpers
def draw_label_bg(frame, text, origin, font, scale,
                  thickness, bg_color, padding=5):
    # Measure the text dimensions so we can draw a perfectly sized background box
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin

    # Draw filled rectangle behind the text
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)

    # Draw the text on top of the rectangle in white
    cv2.putText(frame, text, (x, y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


#Frame Annotator
def annotate_frame(frame, detections, speed_band,
                   speed_source="flow", actual_kmh=None,
                   frame_count=None, fps=None, is_image=False):
    h, w    = frame.shape[:2]
    overall = "LOW"   # tracks the worst severity seen in this frame
    font    = cv2.FONT_HERSHEY_SIMPLEX

    #Per-detection annotations
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        sev   = det["severity"]
        color = SEV_COLOR[sev]

        # HIGH severity gets a thicker box (3px) to stand out more
        thick = 3 if sev == "HIGH" else 2

        # Draw the bounding box around the pothole
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Calculate the safe crossing speed for this specific pothole
        safe  = calc_safe_crossing_speed(
            det["depth_score"], det["bbox"], frame.shape)

        # Build the top label: track ID, severity, confidence, safe speed
        tid   = det.get("track_id", "")
        tid_s = f"#{tid} " if tid != "" else ""
        label = f"{tid_s}[{sev}]  {det['conf']:.2f}  safe:{safe}km/h"
        draw_label_bg(frame, label, (x1, y1 - 6), font, 0.48, 1, color)

        # Draw depth score below the bounding box
        dl = f"Depth: {det['depth_score']:.3f}"
        (dw, dh), _ = cv2.getTextSize(dl, font, 0.45, 1)
        cv2.rectangle(frame, (x1, y2),
                      (x1 + dw + 8, y2 + dh + 8), (30, 30, 30), -1)
        cv2.putText(frame, dl, (x1 + 4, y2 + dh + 2),
                    font, 0.45, color, 1, cv2.LINE_AA)

        # Update overall worst severity for this frame
        if SEVERITY_RANK[sev] > SEVERITY_RANK[overall]:
            overall = sev

    #Calculate speed advisory for this frame 
    rec_speed, urgency = recommended_speed(
        detections, speed_band, frame.shape)
    banner_msg = advisory_message(rec_speed, urgency)
    banner_col = advisory_color(rec_speed, urgency)

    # Top banner (speed advisory)
    # Semi-transparent overlay: copy frame, draw solid rectangle, blend 85/15
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), banner_col, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, banner_msg, (14, 36),
                font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    #Bottom bar (stats)
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    # Thin divider line between frame content and bottom bar
    cv2.line(frame, (0, h - 44), (w, h - 44), (80, 80, 80), 1)

    # Left: total potholes detected in this frame
    cv2.putText(frame, f"Potholes: {len(detections)}", (14, h - 14),
                font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Centre: overall severity (colour coded)
    st = f"Severity: {overall}"
    (sw, _), _ = cv2.getTextSize(st, font, 0.58, 1)
    cv2.putText(frame, st, (w // 2 - sw // 2, h - 14),
                font, 0.58, SEV_COLOR[overall], 1, cv2.LINE_AA)

    # Right: vehicle speed source and frame number
    # Shows GPS speed if OCR found it, otherwise shows optical flow band
    if is_image:
        right_text = "IMAGE MODE"
        right_col  = (180, 180, 180)
    elif actual_kmh is not None:
        # OCR successfully read a GPS speed from the dashcam overlay
        right_text = f"GPS: {actual_kmh} km/h ({speed_band})  |  F:{frame_count}"
        right_col  = SPEED_BAND_COLOR[speed_band]
    else:
        # Falling back to optical flow estimation
        right_text = f"Vehicle: {speed_band} [flow]  |  F:{frame_count}"
        right_col  = SPEED_BAND_COLOR[speed_band]

    (rtw, _), _ = cv2.getTextSize(right_text, font, 0.48, 1)
    cv2.putText(frame, right_text, (w - rtw - 14, h - 14),
                font, 0.48, right_col, 1, cv2.LINE_AA)

    return frame, overall, rec_speed


#Image Pipeline 
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

        # Read the image file into a numpy array (BGR format)
        frame = cv2.imread(img_path)
        if frame is None:
            log_fn(f"ERROR: Cannot read image: {img_path}")
            done_fn(None)
            return

        log_fn(f"Processing : {os.path.basename(img_path)}")
        log_fn(f"Resolution : {frame.shape[1]}x{frame.shape[0]}")
        progress_fn(60)

        # Run YOLO detection 
        results    = yolo(frame, conf=CONFIDENCE, verbose=False)[0]
        detections = []

        if results.boxes is not None and len(results.boxes):
            # Only run depth estimation if at least one pothole was detected (saves time)
            depth_map = estimate_depth(frame, depth_model, depth_transform)

            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2] pixel coords
                conf = float(box.conf[0])           # confidence score (0-1)
                ds   = get_depth_score(depth_map, bbox, frame.shape)  # relative depth
                sev  = classify_severity(ds, bbox, frame.shape)
                detections.append({
                    "bbox": bbox, "conf": conf,
                    "depth_score": ds, "severity": sev,
                })

        progress_fn(80)

        # For images, default to MODERATE speed 
        annotated, overall, rec_spd = annotate_frame(
            frame, detections, "MODERATE", is_image=True)

        # Build timestamped output filename to avoid overwriting previous results
        base      = os.path.splitext(os.path.basename(img_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{base}_pitsense_{timestamp}.png"
        out_path  = os.path.join(OUTPUT_DIR, out_name)

        # Save as PNG
        cv2.imwrite(out_path, annotated)
        progress_fn(100)

        # Print per-detection summary to the GUI log
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


#Video Pipeline
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

        # Open the video file for frame-by-frame reading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_fn(f"ERROR: Cannot open video: {video_path}")
            done_fn(None)
            return

        # Read video metadata
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Build timestamped output filename so previous outputs are never overwritten
        base      = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{base}_pitsense_{timestamp}.mp4"
        out_path  = os.path.join(OUTPUT_DIR, out_name)

        # Set up video writer
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

        # SpeedEstimator maintains its own history across frames
        speed_estimator = SpeedEstimator(history=15)

        # track_history stores the depth score history per ByteTrack ID
        # so each pothole can be smoothed independently across frames
        track_history   = {}

        # OCR caching
        cached_kmh      = None
        OCR_INTERVAL    = 10   # run OCR every 10 frames

        #Main frame loop
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video or read error — exit loop
                break

            frame_count += 1

            # Update optical flow
            flow_band = speed_estimator.update(frame)

            # Start with cached OCR value from last successful read
            actual_kmh   = cached_kmh
            speed_source = "flow"

            # Run OCR every OCR_INTERVAL frames to save processing time
            if ocr.ready and frame_count % OCR_INTERVAL == 0:
                result = ocr.read(frame)
                if result is not None:
                    cached_kmh = result   # update cache
                    actual_kmh = result

            # Decide which speed source to use:
            # OCR is preferred (more accurate), optical flow is the fallback
            if actual_kmh is not None:
                speed_band   = ocr.kmh_to_band(actual_kmh)
                speed_source = "ocr"
            else:
                speed_band = flow_band

            # Run YOLO + ByteTrack detection
            # persist=True keeps track IDs consistent across frames
            results    = yolo.track(frame, conf=CONFIDENCE, persist=True,
                                    tracker="bytetrack.yaml",
                                    verbose=False)[0]
            detections = []

            if results.boxes is not None and len(results.boxes):
                # Only run depth model if potholes were detected (saves time)
                depth_map = estimate_depth(
                    frame, depth_model, depth_transform)

                for box in results.boxes:
                    # Skip boxes where ByteTrack hasn't assigned an ID yet
                    # (happens on the very first frame a pothole appears)
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    bbox     = box.xyxy[0].cpu().numpy()
                    conf     = float(box.conf[0])

                    # Get raw depth score for this detection
                    ds       = get_depth_score(
                        depth_map, bbox, frame.shape)

                    # Add to this track's depth history (creates new deque if first time)
                    if track_id not in track_history:
                        track_history[track_id] = deque(
                            maxlen=SMOOTH_WINDOW)
                    track_history[track_id].append(ds)

                    # Use smoothed depth score (average over last N frames)
                    # Reduces flickering caused by single-frame depth estimation noise
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

            # Annotate the frame and write it to the output video
            annotated, _, rec_spd = annotate_frame(
                frame, detections, speed_band,
                speed_source=speed_source,
                actual_kmh=actual_kmh,
                frame_count=frame_count,
                fps=vid_fps,
                is_image=False)
            out.write(annotated)

            # Update the GUI progress bar
            pct = int((frame_count / total) * 100)
            progress_fn(pct)

            # Log a status update every 60 frames (every ~2 seconds at 30fps)
            if frame_count % 60 == 0:
                spd_str  = f"{rec_spd} km/h" if rec_spd else "safe"
                src_str  = f"GPS:{actual_kmh}km/h" if actual_kmh else f"flow:{speed_band}"
                log_fn(f"  Frame {frame_count}/{total} ({pct}%)"
                       f"  |  Speed: {src_str}"
                       f"  |  Advisory: {spd_str}"
                       f"  |  {len(detections)} pothole(s)")

        # Release file handles after processing is complete
        cap.release()
        out.release()

        # Print final summary to the GUI log
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


#Pipeline Router
def run_pipeline(file_path, log_fn, progress_fn, done_fn):
    # Decide which pipeline to run based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        process_image(file_path, log_fn, progress_fn, done_fn)
    elif ext in VIDEO_EXTS:
        process_video(file_path, log_fn, progress_fn, done_fn)
    else:
        log_fn(f"ERROR: Unsupported file type '{ext}'")
        done_fn(None)


#GUI
class PitSenseApp:
    def __init__(self, root):
        self.root      = root
        self.file_path = None   # stores the currently selected file path
        self.root.title("PitSense — Pothole Detection System")
        self.root.geometry("700x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f1117")
        self._build_ui()

        # Show dependency error in the log if any import failed
        if not DEPS_OK:
            self.log(f"Missing dependency: {DEPS_ERROR}")
            self.log("Make sure your pothole_env is activated.")

    def _build_ui(self):
        # Colour palette
        bg   = "#0f1117"   # dark background
        card = "#1a1d27"   # slightly lighter card background
        acc  = "#e63946"   # red accent colour
        sub  = "#8d99ae"   # subdued text colour

        #Header bar
        hdr = tk.Frame(self.root, bg=acc, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="PitSense",
                 font=("Helvetica", 20, "bold"),
                 bg=acc, fg="white").pack(side="left", padx=18, pady=10)
        tk.Label(hdr, text="Pothole Detection & Speed Advisory",
                 font=("Helvetica", 10),
                 bg=acc, fg="#ffcdd2").pack(side="left", pady=10)

        # OCR status badge 
        ocr_color = "#2d6a4f" if OCR_AVAILABLE else "#555"
        ocr_text  = "OCR ON" if OCR_AVAILABLE else "OCR OFF"
        tk.Label(hdr, text=ocr_text,
                 font=("Helvetica", 8, "bold"),
                 bg=ocr_color, fg="white",
                 padx=6, pady=3).pack(side="right", padx=12)

        #File picker row
        drop = tk.Frame(self.root, bg=card, highlightthickness=2,
                        highlightbackground="#2d3250")
        drop.pack(fill="x", padx=20, pady=(16, 0))
        self.file_label = tk.Label(
            drop, text="No file selected",
            font=("Helvetica", 10), bg=card, fg=sub, anchor="w")
        self.file_label.pack(side="left", padx=14, pady=12,
                             fill="x", expand=True)

        # Browse Image button 
        tk.Button(drop, text="Browse Image",
                  font=("Helvetica", 9, "bold"),
                  bg="#2d6a4f", fg="white", relief="flat",
                  padx=10, pady=6, cursor="hand2",
                  activebackground="#1b4332",
                  activeforeground="white",
                  command=self.browse_image).pack(
                  side="right", padx=(4, 10), pady=8)

        # Browse Video button 
        tk.Button(drop, text="Browse Video",
                  font=("Helvetica", 9, "bold"),
                  bg=acc, fg="white", relief="flat",
                  padx=10, pady=6, cursor="hand2",
                  activebackground="#c1121f",
                  activeforeground="white",
                  command=self.browse_video).pack(
                  side="right", padx=0, pady=8)

        # Mode info label
        self.type_label = tk.Label(
            self.root, text="",
            font=("Helvetica", 9), bg=bg, fg=sub)
        self.type_label.pack(anchor="e", padx=22)

        #Progress bar
        pf = tk.Frame(self.root, bg=bg)
        pf.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(pf, text="Progress", font=("Helvetica", 9),
                 bg=bg, fg=sub).pack(anchor="w")

        # Style the progress bar to match the app colour scheme
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

        #Action buttons 
        btn_row = tk.Frame(self.root, bg=bg)
        btn_row.pack(pady=(12, 0))

        # Run button 
        self.run_btn = tk.Button(
            btn_row, text="Run PitSense",
            font=("Helvetica", 12, "bold"),
            bg=acc, fg="white", relief="flat",
            padx=20, pady=10, cursor="hand2",
            activebackground="#c1121f", activeforeground="white",
            state="disabled", command=self.start_processing)
        self.run_btn.pack(side="left", padx=(0, 14))

        # Open output folder button 
        self.open_btn = tk.Button(
            btn_row, text="Open Output Folder",
            font=("Helvetica", 11),
            bg="#457b9d", fg="white", relief="flat",
            padx=16, pady=10, cursor="hand2",
            activebackground="#1d6c8a", activeforeground="white",
            command=self.open_output_folder)
        self.open_btn.pack(side="left")

        #Processing log 
        lf = tk.Frame(self.root, bg=bg)
        lf.pack(fill="both", expand=True, padx=20, pady=(12, 0))
        tk.Label(lf, text="Processing Log",
                 font=("Helvetica", 9), bg=bg, fg=sub).pack(anchor="w")

        # Read-only text box 
        self.log_box = tk.Text(
            lf, bg=card, fg="#a8dadc",
            font=("Courier", 9), relief="flat",
            state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        #Footer 
        ocr_note = "OCR speed reading enabled" if OCR_AVAILABLE \
                   else "Install easyocr for GPS speed reading"
        tk.Label(self.root,
                 text=f"PitSense  •  Internship Project  •  {ocr_note}",
                 font=("Helvetica", 8), bg=bg,
                 fg="#3d405b").pack(pady=(6, 8))

    def browse_video(self):
        # Open file dialog filtered to video formats
        path = filedialog.askopenfilename(
            title="Select a road video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")])
        if path:
            self._set_file(path, "VIDEO")

    def browse_image(self):
        # Open file dialog filtered to image formats
        path = filedialog.askopenfilename(
            title="Select a road image",
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")])
        if path:
            self._set_file(path, "IMAGE")

    def _set_file(self, path, kind):
        # Store file path and update the UI to reflect the selection
        self.file_path = path
        name = os.path.basename(path)
        size = os.path.getsize(path) / (1024 * 1024)   # convert bytes to MB
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

        # Enable the Run button now that a file is selected
        self.run_btn.config(state="normal")
        self.log(f"Selected [{kind}]: {path}")

    def log(self, msg):
        # Temporarily enable the text box to insert a line, then disable again
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")   # auto-scroll to the latest line
        self.log_box.config(state="disabled")

    def set_progress(self, pct):
        # Update the progress bar and percentage label
        self.progress["value"] = pct
        self.pct_label.config(text=f"{pct}%")
        self.root.update_idletasks()   # force GUI refresh immediately

    def start_processing(self):
        if not self.file_path or not DEPS_OK:
            return

        # Disable run button and reset progress while processing
        self.run_btn.config(state="disabled", text="Processing...")
        self.progress["value"] = 0

        # Run the pipeline in a background thread so the GUI stays responsive
        # lambda wrappers schedule GUI updates on the main thread via root.after()
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
        # Re-enable the run button when processing finishes
        self.run_btn.config(state="normal", text="Run PitSense")
        if out_path:
            self.set_progress(100)
            self.log("\nOutput ready — click 'Open Output Folder' to view.")
        else:
            self.log("Processing failed. Check the log above.")

    def open_output_folder(self):
        # Create the output folder if it doesn't exist, then open it in Explorer
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.startfile(OUTPUT_DIR)


# Entry point 
if __name__ == "__main__":
    root = tk.Tk()
    app  = PitSenseApp(root)
    root.mainloop()   # starts the GUI event loop 