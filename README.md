# YOLOv5 Face Detection Demo

This project provides a simplified example of using YOLOv5 to detect faces in images, videos and webcam streams.

## Requirements

* **Python**: 3.8 or newer is recommended.
* Install Python packages:

```bash
pip install -r requirements.txt
```

## Downloading the Weights

The detector expects the weight file `yolov5s-face.pt` in the project root. If you do not have it yet, download it from the [YOLOv5-face repository](https://github.com/deepcam-cn/yolov5-face) releases page and place the file next to `detect_simple.py`.

## Running the Demo

Use the `--source` argument of `detect_simple.py` to specify an input image, video file or webcam index.

```bash
# Image
python detect_simple.py --source path/to/image.jpg

# Video
python detect_simple.py --source path/to/video.mp4

# Webcam (device 0)
python detect_simple.py --source 0
```

The results are displayed in a window. Press `ESC` to close.

## Troubleshooting

* **FileNotFoundError: Model file not found: yolov5s-face.pt**
  - Ensure you have downloaded `yolov5s-face.pt` and placed it in this directory.
* **cv2 errors or unable to open video/webcam**
  - Make sure the `opencv-python` package is installed. Some platforms also require additional system dependencies (e.g. `ffmpeg`, `libGL`).

