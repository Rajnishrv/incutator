import cv2
import numpy as np

# Load the video (background or animation)
video = cv2.VideoCapture("km_20221218_1080p.mp4")

# Use webcam to track ball hits
cap = cv2.VideoCapture(0)

# Initial shape parameters for the object
object_pos = [300, 200]  # x, y
object_radius = 40
shape_changed = False

def detect_ball(frame):
    """Detect a thrown ball using color filtering (tune HSV as needed)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 100])  # Example for red ball
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_center = None
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            ball_center = (int(x), int(y))
            break
    return ball_center

while True:
    ret_video, video_frame = video.read()
    ret_cam, cam_frame = cap.read()

    if not ret_video or not ret_cam:
        break

    cam_frame = cv2.flip(cam_frame, 1)  # Flip for mirror effect

    # Detect the ball
    ball = detect_ball(cam_frame)

    # Draw game object on video frame
    if not shape_changed:
        cv2.circle(video_frame, tuple(object_pos), object_radius, (0, 255, 0), -1)
    else:
        cv2.rectangle(video_frame, (object_pos[0] - 40, object_pos[1] - 40),
                      (object_pos[0] + 40, object_pos[1] + 40), (0, 0, 255), -1)

    # Detect collision
    if ball:
        distance = np.linalg.norm(np.array(ball) - np.array(object_pos))
        if distance < object_radius + 20:
            shape_changed = True

        # Show ball position on webcam feed
        cv2.circle(cam_frame, ball, 10, (255, 255, 255), -1)

    # Display frames
    cv2.imshow("Projected Video", video_frame)
    cv2.imshow("Webcam Ball Detection", cam_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
