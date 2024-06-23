import os
os.system("python yolov5/train.py --img 416 --batch 16 --epochs 20 --data yolov5/data.yaml --weights yolov5s.pt --name spfd --cache")