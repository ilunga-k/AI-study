import torch
import cv2
import numpy as np
from pathlib import Path
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

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img0 = frame.copy()
    img = letterbox(img0, imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for det in pred:
        if len(det):
            # 좌표 스케일 복원
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            det[:, 5:15] = scale_coords_landmarks(
                img.shape[2:], det[:, 5:15], img0.shape
            ).round()

            for d in det:
                xyxy = d[0:4]
                conf = d[4]
                landmarks = d[5:15]
                cls = d[15] if d.shape[0] > 15 else None

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for i in range(5):
                    cx = int(landmarks[2 * i])
                    cy = int(landmarks[2 * i + 1])
                    cv2.circle(img0, (cx, cy), 2, (255, 0, 0), -1)

    # 결과 출력
    cv2.imshow("YOLOv5-face simplified", img0)
    if cv2.waitKey(1) == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
