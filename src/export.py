from ultralytics import YOLO

model = YOLO("./weights/best.pt")

model.export(format="onnx") 
model.export(format="engine", half=True) #TensorRT FP16