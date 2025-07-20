import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Avatar parameters
AVATAR_COLOR = (0, 0, 255)  # Red color (BGR)
AVATAR_THICKNESS = 5
AVATAR_SCALE = 2.5  # Scale factor for avatar size
AVATAR_POSITION = (50, 50)   # Top-left position of avatar

cap = cv2.VideoCapture(0)  # Open webcam

def draw_avatar(image, landmarks):
    """Draw a simplified stick figure avatar based on pose landmarks"""
    h, w = image.shape[:2]
    scale = min(h, w) / AVATAR_SCALE
    
    # Helper function to convert normalized landmarks to avatar coordinates
    def get_point(landmark):
        x = int(AVATAR_POSITION[0] + landmark.x * scale)
        y = int(AVATAR_POSITION[1] + landmark.y * scale)
        return (x, y)
    
    try:
        # Head (circle)
        head = get_point(landmarks[mp_pose.PoseLandmark.NOSE])
        cv2.circle(image, head, int(15 * scale/100), AVATAR_COLOR, -1)

        # Body connections (torso, arms, legs)
        connections = [
            # Torso
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            
            # Arms
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            
            # Legs
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Draw all connections
        for connection in connections:
            start = get_point(landmarks[connection[0]])
            end = get_point(landmarks[connection[1]])
            cv2.line(image, start, end, AVATAR_COLOR, AVATAR_THICKNESS)
            
    except (IndexError, AttributeError):
        pass  # Skip if any landmarks are missing

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for selfie view and convert color
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get body pose landmarks
        results = pose.process(image_rgb)

        # Draw landmarks on original frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            
            # Draw avatar performing same pose
            draw_avatar(frame, results.pose_landmarks.landmark)

        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()