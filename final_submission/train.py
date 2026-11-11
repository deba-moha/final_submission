from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    results = model.train(
        data="dataset/data.yaml", 
        epochs=45, 
        imgsz=640, 
        batch=16, 
        device=0
    )

if __name__ == "__main__":
    main()
