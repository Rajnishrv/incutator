import cv2
import numpy as np

# Initialize
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or video path
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Tracking and counting
tracker = {}
track_id = 0
max_disappeared = 20
in_count = 0
out_count = 0

# Line position (horizontal line at 2/3 height)
H, W = None, None
LINE_Y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if W is None or H is None:
        H, W = frame.shape[:2]
        LINE_Y = int(2 * H / 3)  # Counting line position
    
    # Preprocessing
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 800:  # Min size threshold
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        current_centroids.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Update tracker
    disappeared_ids = list(tracker.keys())
    
    for cx, cy in current_centroids:
        min_dist = float('inf')
        min_id = None
        
        for obj_id, (prev_cx, prev_cy, disappeared) in tracker.items():
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            
            if dist < 50 and dist < min_dist:  # Matching threshold
                min_dist = dist
                min_id = obj_id
        
        if min_id is not None:
            # Update existing track
            tracker[min_id] = (cx, cy, 0)
            if min_id in disappeared_ids:
                disappeared_ids.remove(min_id)
        else:
            # New object detected
            track_id += 1
            tracker[track_id] = (cx, cy, 0)
    
    # Handle disappeared tracks
    for obj_id in disappeared_ids:
        cx, cy, disappeared = tracker[obj_id]
        disappeared += 1
        
        if disappeared >= max_disappeared:
            tracker.pop(obj_id)
        else:
            tracker[obj_id] = (cx, cy, disappeared)
    
    # Line crossing logic
    for obj_id, (cx, cy, _) in list(tracker.items()):
        if obj_id not in tracker:
            continue
        
        prev_cy = tracker[obj_id][1]  # Previous y-position
        
        # Entry (moving up: current above line, previous below)
        if cy < LINE_Y and prev_cy >= LINE_Y:
            in_count += 1
            tracker.pop(obj_id)  # Remove after counting
        
        # Exit (moving down: current below line, previous above)
        elif cy >= LINE_Y and prev_cy < LINE_Y:
            out_count += 1
            tracker.pop(obj_id)  # Remove after counting
    
    # Draw UI elements
    cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)
    inside_count = max(0, in_count - out_count)
    cv2.putText(frame, f"Entries: {in_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {out_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Inside: {inside_count}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow("People Counter", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()