import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (uses pretrained coco model)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open camera or video
cap = cv2.VideoCapture(0)  # Use file path instead of 0 for video file

line_position = 250
people_in = 0
people_out = 0
track_memory = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml")
    
    detections = []
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Store previous y position
            if track_id not in track_memory:
                track_memory[track_id] = cy
            else:
                prev_cy = track_memory[track_id]
                if prev_cy < line_position and cy >= line_position:
                    people_in += 1
                    track_memory[track_id] = cy
                elif prev_cy > line_position and cy <= line_position:
                    people_out += 1
                    track_memory[track_id] = cy

    # Draw counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f"IN: {people_in}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {people_out}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
