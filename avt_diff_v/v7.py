import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load cartoon head image (with alpha channel)
funny_head = cv2.imread("Image2.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def overlay_transparent(background, overlay, x, y, scale=1.0):
    if overlay is None:
        return background

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w = overlay.shape[:2]

    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a / 255.0

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask) + overlay_rgb[:, :, c] * mask

    return background

def draw_funny_avatar(image, landmarks):
    h, w = image.shape[:2]

    try:
        # Head position
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        x = int(nose.x * w) - funny_head.shape[1] // 4
        y = int(nose.y * h) - funny_head.shape[0] // 2

        # Apply funny head image
        image = overlay_transparent(image, funny_head, x, y, scale=0.4)

    except Exception as e:
        print("Error:", e)

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
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = draw_funny_avatar(frame, results.pose_landmarks.landmark)

        cv2.imshow("Funny Avatar Pose", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
