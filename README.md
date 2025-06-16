#  YOLOv5 + SORT Object Detection and Tracking

This project demonstrates real-time **object detection and tracking** using the **YOLOv5** model and a minimal implementation of the **SORT (Simple Online and Realtime Tracking)** algorithm. It captures video from a webcam, detects objects frame-by-frame, and tracks them by assigning persistent IDs.

---

##  Technologies Used

- **YOLOv5** (via PyTorch Hub)
- **OpenCV** for video capture and visualization
- **NumPy** for matrix operations
- **SORT** for lightweight object tracking
- **Kalman Filter** from `filterpy`

---

## Features

- Real-time object detection using YOLOv5
- Simple tracking with temporary SORT implementation
- Assigns unique IDs to objects across frames
- Filters detections based on confidence threshold
- Displays bounding boxes and object IDs on live video

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yolo-sort-tracker.git
cd yolo-sort-tracker

2. Install Requirements
pip install torch torchvision opencv-python filterpy numpy

Run the App
python your_script_name.py
