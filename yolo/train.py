from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=r"C:\workspace\Behavior-Data-Analysis-System-Based-on-Deep-Learning\data\skeletons",
        epochs=150,
        imgsz=224,
        batch=128,
        device=0,
        optimizer="AdamW",
        lr0=1e-3,
        patience=30,
        project="runs/cls",
        name="skeletons",
        workers=8
    )

if __name__ == "__main__":
    main()
