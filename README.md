# Fire & Smoke Detection on Edge Devices

**Work in progress**

Real-time fire and smoke detection using YOLO, optimized for
edge inference on NVIDIA Jetson. Focused on reducing false
positives from common confusers like sunset glare and warm lighting.

## Planned Approach
- YOLOv8 trained on the D-Fire dataset
- Hard-negative mining to reduce false positives
- TensorRT optimization (FP16/INT8) for edge deployment
- Real-time video demo

## Status
Currently setting up the dataset and training pipeline.
Results, benchmarks, and demo will be added as the project develops.
