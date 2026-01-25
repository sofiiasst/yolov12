from ultralytics import YOLO

# Use YAML model (nc: 1 for 1 class)
model = YOLO("ultralytics/cfg/models/v12/yolov12n.yaml")

result = model.train(
    data="ultralytics/cfg/datasets/tool.yaml",
    epochs=15,
    imgsz=640,
    batch=1,
    lr0=0.01,
    device='cpu',  # Change to 0 for GPU on Colab
    project="runs",
    name="training",
    save_json=True,
    save_txt=True,
    save_conf=True,
    exist_ok=False,
    patience=0,
)