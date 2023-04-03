#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import os 


_DATASET_DIR = "/home/pcms/"
_DATASET_NAME = "my_dataset"
_DATASET_YAML = _DATASET_NAME + ".yaml"

root = os.path.join(_DATASET_DIR, _DATASET_NAME)
yaml = os.path.join(root, _DATASET_YAML)

model = YOLO('yolov8n.pt')
results = model.train(
	data = yaml,
	imgsz = 640,
	epochs = 500,
	batch = 8,
	name = 'custom'
)

path = "%s/runs/detect/custom/weights/best.pt" % _DATASET_DIR
det_model = YOLO(path)
det_model.export(format="openvino", dynamic=True, half=False)
