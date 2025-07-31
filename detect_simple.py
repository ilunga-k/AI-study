import argparse
import os

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import (
    non_max_suppression_face,
    scale_coords,
    scale_coords_landmarks,
)
from utils.datasets import letterbox

# 설정
weights = "yolov5s-face.pt"  # 모델 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsz = 640
conf_thres = 0.5
iou_thres = 0.3

# 모델 로드
model = attempt_load(weights, map_location=device)
model.eval()


def detect(img0):
    """Run detection on a single image."""
    img = letterbox(img0, imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape
            ).round()
            det[:, 5:15] = scale_coords_landmarks(
                img.shape[2:], det[:, 5:15], img0.shape
            ).round()

            for d in det:
                xyxy = d[0:4]
                landmarks = d[5:15]

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for i in range(5):
                    cx = int(landmarks[2 * i])
                    cy = int(landmarks[2 * i + 1])
                    cv2.circle(img0, (cx, cy), 2, (255, 0, 0), -1)
    return img0


def process_video(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img0 = detect(frame.copy())
        cv2.imshow("YOLOv5-face simplified", img0)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


def process_image(path):
    img0 = cv2.imread(path)
    if img0 is None:
        print(f"Image {path} not found")
        return
    img0 = detect(img0)
    cv2.imshow("YOLOv5-face simplified", img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser(description="Simple face detection demo")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="image path, video path or webcam index",
    )
    return parser.parse_args()


def main(opt):
    source = opt.source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        process_video(cap)
    elif os.path.splitext(source)[1].lower() in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ]:
        process_image(source)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open {source}")
            return
        process_video(cap)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
