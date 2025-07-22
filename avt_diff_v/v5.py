import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load funny images with transparency (ensure these files exist)
funny_head = cv2.imread("funny_head.png", cv2.IMREAD_UNCHANGED)  # with alpha
funny_arm = cv2.imread("funny_arm.png", cv2.IMREAD_UNCHANGED)
funny_leg = cv2.imread("funny_leg.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)  # Open webcam

def overlay_transparent(background, overlay, x, y, scale=1.0):
    """Overlay transparent PNG onto background at (x, y) with optional scale"""
    if overlay is None:
        return background

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w = overlay.shape[:2]

    # Ensure placement is within frame
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]

    # Split overlay into color and alpha
    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r))
    mask = a / 255.0

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1.0 - mask) + overlay_color[:, :, c] * mask
    return background

def draw_funny_avatar(image, landmarks):
    """Draw funny avatar parts at specific body landmarks"""
    h, w = image.shape[:2]

    try:
        # Head: Place at NOSE
        head = landmarks[mp_pose.PoseLandmark.NOSE]
        x = int(head.x * w) - funny_head.shape[1] // 4
        y = int(head.y * h) - funny_head.shape[0] // 2
        image = overlay_transparent(image, funny_head, x, y, scale=0.5)

        # Left Arm: Place between SHOULDER and ELBOW
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        x = int(left_shoulder.x * w)
        y = int(left_shoulder.y * h)
        image = overlay_transparent(image, funny_arm, x - 30, y, scale=0.4)

        # Right Arm
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x = int(right_shoulder.x * w)
        y = int(right_shoulder.y * h)
        image = overlay_transparent(image, funny_arm, x - 30, y, scale=0.4)

        # Left Leg: Place at LEFT_HIP
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        x = int(left_hip.x * w)
        y = int(left_hip.y * h)
        image = overlay_transparent(image, funny_leg, x - 20, y, scale=0.5)

        # Right Leg
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        x = int(right_hip.x * w)
        y = int(right_hip.y * h)
        image = overlay_transparent(image, funny_leg, x - 20, y, scale=0.5)

    except Exception as e:
        print("Error drawing funny avatar:", e)

    return image

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            # Apply funny avatar
            frame = draw_funny_avatar(frame, results.pose_landmarks.landmark)

        cv2.imshow('Funny Avatar Pose Tracker', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
