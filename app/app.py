import gradio as gr
from ultralytics import YOLO

model = YOLO("./weights/best.pt")

def detect(img):
    results = model.predict(img, conf=0.35)  # F1 peaked at 0.359
    return results[0].plot()

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Fire & Smoke Detection (YOLO11s)",
    description="Upload an image to detect fire and smoke."
)
demo.launch()