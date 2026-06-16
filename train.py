from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\Projects\BharatPotHole\BharatPotHole\BharatPotHole\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="pothole_v1",
        project=r"C:\Projects\runs",
        workers=0
    )

    print("Training complete! Model saved to C:\\Projects\\runs\\pothole_v1\\weights\\best.pt")