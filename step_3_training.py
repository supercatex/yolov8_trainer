#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import os 
import yaml 
import shutil


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

dir_data = os.path.join(cfg["dir"]["root"], cfg["dir"]["data"])
uri_yaml = os.path.join(dir_data, cfg["dir"]["data"] + ".yaml")

shutil.rmtree("./runs/detect", ignore_errors=False, onerror=None)
os.removedirs("./runs")

model = YOLO('yolov8n.pt')
# model.to("cuda")
results = model.train(
	data = uri_yaml,
	imgsz = 640,
	epochs = 500,
	batch = 8,
	name = 'custom'
)

path = "./runs/detect/custom/weights/best.pt"
det_model = YOLO(path)
det_model.export(format="openvino", dynamic=True, half=False)
