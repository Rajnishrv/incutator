import cv2
import mediapipe as mp
import socket
import json

# Set up socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
blender_address = ("127.0.0.1", 5005)

# Init MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Select a few landmark points (you can expand)
        data = {
            'left_shoulder': {
                'x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                'y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            },
            'right_shoulder': {
                'x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                'y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            },
            'left_elbow': {
                'x': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                'y': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
            },
            'right_elbow': {
                'x': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                'y': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            }
        }

        json_data = json.dumps(data)
        sock.sendto(json_data.encode(), blender_address)

    cv2.imshow("Pose Sender", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
sock.close()
cv2.destroyAllWindows()
