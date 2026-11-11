# Model Mayhem - PCB Defect Detection

TEAM _ BANANA
Members _ Debasis Mohapatra



## Setup
Install the required dependencies:
```bash
pip install -r requirements.txt

```

## Training

Ensure your dataset is located in a `dataset` folder with a properly configured `data.yaml` file.
Run the training script:

```bash
python train.py

```

*Note: The model was trained for 45 epochs using YOLOv8s with a 640px image resolution.*

## Inference

1. Place all unseen test images into a folder named `test_images` in the same directory as the script.
2. Ensure the trained model weights file (`best.pt`) is in the same directory.
3. Run the inference script:

```bash
python inference.py

```

*Note: The inference script is optimized to process images one-by-one at 640px resolution to ensure stable memory usage on standard GPUs.*

## Output

The script will generate a `predictions.csv` file in the root directory, perfectly formatted to the prescribed competition requirements (image_name, class_name, x_min, y_min, x_max, y_max, confidence).

```

```
