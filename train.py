from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  

    model.train(
        data=r"C:\Projects\PitSense\BharatPotHole\BharatPotHole\BharatPotHole\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        name="pothole_v2",
        project=r"C:\Projects\PitSense\runs",

        # ── Optimizer ────────────────────────────────────────
        optimizer="AdamW",      
        lr0=0.001,              
        lrf=0.01,               
        warmup_epochs=5,        

        # ── Regularization ───────────────────────────────────
        weight_decay=0.0005,
        dropout=0.1,            

        # ── Augmentation ─────────────────────────────────────
        hsv_h=0.015,            
        hsv_s=0.7,              
        hsv_v=0.4,              
        flipud=0.0,            
        fliplr=0.5,             
        mosaic=1.0,            
        mixup=0.1,             
        degrees=5.0,            
        translate=0.1,         
        scale=0.5,              

        # ── Early stopping ───────────────────────────────────
        patience=20,            
        save_period=25,        

        # ── Logging ──────────────────────────────────────────
        plots=True,            
        verbose=True,
    )

    print("\nTraining complete!")
    print("Best model saved to: C:\\Projects\\PitSense\\runs\\pothole_v2\\weights\\best.pt")
    print("Check training plots at: C:\\Projects\\PitSense\\runs\\pothole_v2\\")