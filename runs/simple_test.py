from ultralytics import YOLO
from pathlib import Path

# Load latest trained model
training_folders = sorted(Path("runs").glob("training*"), key=lambda p: int(p.name.replace("training", "") or 0), reverse=True)
if not training_folders:
    raise FileNotFoundError("No training folders found. Train first with: python runs/train.py")

model_path = training_folders[0] / "weights" / "best.pt"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

model = YOLO(str(model_path))

# Run inference on dataset
results = model('datasets/tool/images/train/', conf=0.01, max_det=1)

# Show the results
results[1].show()
