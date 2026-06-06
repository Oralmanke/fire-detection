from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Control the cuda if you are using Nvidia
    print(torch.cuda.is_available())

    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        patience = 20,
        imgsz=960,
        batch=8,
        project="runs",
        name="SecondRun",
        device=0,
    )
