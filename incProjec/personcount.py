import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (you can also use 'yolov8n.pt' or 'yolov8s.pt' etc.)
model = YOLO("yolov8n.pt")  # Small and fast

# Video capture (0 = webcam or give file path)
cap = cv2.VideoCapture(0)

# Define line position and counts
line_position = 300
offset = 6  # tolerance
people_in = 0
people_out = 0

# Store tracked object IDs and their previous positions
track_history = {}

# Tracker from OpenCV (Simple tracker)
tracker = cv2.TrackerKCF_create()

def center(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    new_track_history = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue  # Only detect persons (class 0)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx, cy = center(x1, y1, x2 - x1, y2 - y1)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Track ID (basic using hash of center)
            obj_id = hash((round(cx / 10), round(cy / 10)))

            if obj_id in track_history:
                prev_cx, prev_cy = track_history[obj_id]
                if prev_cy < line_position <= cy:
                    people_in += 1
                elif prev_cy > line_position >= cy:
                    people_out += 1

            new_track_history[obj_id] = (cx, cy)

    track_history = new_track_history.copy()

    # Draw counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 255), 2)
    cv2.putText(frame, f"In: {people_in}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {people_out}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
