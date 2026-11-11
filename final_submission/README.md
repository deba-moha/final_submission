# Model Mayhem - PCB Defect Detection

TEAM _ BANANA
Members _ Debasis Mohapatra

## Setup
Run: pip install -r requirements.txt

## Training
Ensure dataset is in a 'dataset' folder with 'data.yaml' configured, then run:
python train.py

## Inference
Place the unseen test images in a folder named 'test_images'. Ensure 'best.pt' is in the same directory.
Run:
python inference.py
This will run predictions using Test-Time Augmentation (TTA) and upscale images to 800px for better small-defect detection.
It generates 'predictions.csv' in the prescribed competition format.
