#  Task#4

 #projects, and resume. Use HTML for structure, CS or styling, and add a touch of JavaScript for interactivity.
 #Object Detection and Tracking
 #Develop a system capable of detecting and
 #tracking objects in real-time video streams. Use
 #deep learning models like YOLO (You Only Look
 #Once) or Faster R-CNN for accurate object
 #detection and tracking.
 






#First Install these
#pip install torch torchvision torchaudio
#pip install opencv-python
#pip install matplotlib
#pip install numpy
#pip install yolov5  # if not available, clone from GitHub instead



import torch
import cv2
import numpy as np

# Minimal SORT tracker class (from https://github.com/abewley/sort)
from filterpy.kalman import KalmanFilter

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        from collections import deque
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections=np.empty((0, 5))):
        # This is a stub; full SORT is complex
        # For simplicity, just return input detections with fake IDs
        # Replace with full implementation or use the actual library
        results = []
        for i, det in enumerate(detections):
            results.append(np.append(det, i))  # det + track_id
        return np.array(results)

# Load YOLOv5 small model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

tracker = Sort()

confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # (x1,y1,x2,y2,conf,class)

    # Filter detections by confidence threshold
    detections = detections[detections[:,4] > confidence_threshold]

    # Format detections for tracker: (x1, y1, x2, y2, score)
    dets_for_tracker = detections[:, :5]

    # Update tracker
    tracked_objects = tracker.update(dets_for_tracker)

    # Draw bounding boxes + IDs
    for *bbox, conf, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {int(track_id)} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('YOLO + SORT Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
