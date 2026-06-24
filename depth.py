import torch
import cv2
import numpy as np

# Load MiDaS model
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return model, transform

# Get depth score for a single bounding box region
def get_depth_score(depth_map, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Clamp to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop depth map to bounding box
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    # Normalize depth score 0-1
    road_depth = np.mean(depth_map)
    pothole_depth = np.mean(region)
    score = abs(pothole_depth - road_depth) / (road_depth + 1e-6)
    return float(np.clip(score, 0, 1))

# Run depth estimation on a frame
def estimate_depth(frame, model, transform):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

if __name__ == '__main__':
    print("Loading MiDaS depth model...")
    model, transform = load_depth_model()
    print("Depth model loaded successfully!")