import os, glob, gc
import pandas as pd
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")
    test_dir = "test_images"
    image_paths = sorted(glob.glob(f"{test_dir}/*.[jJ][pP][gG]") + glob.glob(f"{test_dir}/*.[pP][nN][gG]"))
    
    print(f"Running bulletproof inference on {len(image_paths)} images...")
    
    rows = []
    classes = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}
    
    # Process strictly one image at a time
    for img_path in image_paths:
        res = model.predict(
            source=img_path, 
            conf=0.20, 
            iou=0.50, 
            imgsz=640,       # Back to safe baseline
            augment=False,   # TTA off for guaranteed stability
            device=0, 
            verbose=False
        )[0]
        
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rows.append({
                "image_name": os.path.basename(res.path),
                "class_name": classes[int(box.cls[0])],
                "x_min": int(x1), 
                "y_min": int(y1), 
                "x_max": int(x2), 
                "y_max": int(y2),
                "confidence": round(float(box.conf[0]), 4)
            })
            
        # Force memory cleanup after every single image
        del res
        
    pd.DataFrame(rows).to_csv("predictions.csv", index=False)
    print("Saved predictions.csv successfully.")

if __name__ == "__main__":
    main()
