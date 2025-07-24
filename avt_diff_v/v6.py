import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load cartoon part images
funny_head = cv2.imread("Image2.png", cv2.IMREAD_UNCHANGED)
funny_arm = cv2.imread("Image3.png", cv2.IMREAD_UNCHANGED)
funny_leg = cv2.imread("Image5.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def overlay_transparent(bg, overlay, x, y, scale=1.0):
    if overlay is None: return bg
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w = overlay.shape[:2]

    if x + w > bg.shape[1]: w = bg.shape[1] - x
    if y + h > bg.shape[0]: h = bg.shape[0] - y
    overlay = overlay[:h, :w]

    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0
    overlay_rgb = cv2.merge((b, g, r))

    for c in range(3):
        bg[y:y+h, x:x+w, c] = bg[y:y+h, x:x+w, c] * (1 - mask) + overlay_rgb[:, :, c] * mask
    return bg

def draw_funny_avatar(image, landmarks):
    h, w = image.shape[:2]
    try:
        # HEAD
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        x = int(nose.x * w) - 40
        y = int(nose.y * h) - 60
        image = overlay_transparent(image, funny_head, x, y, scale=0.4)

        # LEFT ARM
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        x = int(left_shoulder.x * w) - 30
        y = int(left_shoulder.y * h)
        image = overlay_transparent(image, funny_arm, x, y, scale=0.3)

        # RIGHT ARM
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x = int(right_shoulder.x * w) - 30
        y = int(right_shoulder.y * h)
        image = overlay_transparent(image, funny_arm, x, y, scale=0.3)

        # LEFT LEG
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        x = int(left_hip.x * w) - 20
        y = int(left_hip.y * h)
        image = overlay_transparent(image, funny_leg, x, y, scale=0.4)

        # RIGHT LEG
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        x = int(right_hip.x * w) - 20
        y = int(right_hip.y * h)
        image = overlay_transparent(image, funny_leg, x, y, scale=0.4)

    except Exception as e:
        print("Avatar error:", e)

    return image

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = draw_funny_avatar(frame, results.pose_landmarks.landmark)

        cv2.imshow("Funny Cartoon Avatar", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
