import cv2
import numpy as np

# Function to detect fingers
def detect_fingers(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        
        # Convex hull
        hull = cv2.convexHull(max_contour, returnPoints=False)
        
        # Convexity defects
        defects = cv2.convexityDefects(max_contour, hull)
        
        if defects is not None:
            count_fingers = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                
                # Calculate the angle to detect fingers
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
                
                # If the angle is less than 90 degrees, it's a finger
                if angle <= np.pi / 2:
                    count_fingers += 1
                    cv2.circle(frame, far, 8, (255, 0, 0), -1)
            
            # Display the number of fingers
            cv2.putText(frame, f"Fingers: {count_fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect fingers
    frame = detect_fingers(frame)
    
    # Display the frame
    cv2.imshow("Finger Detection", frame)
    
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
