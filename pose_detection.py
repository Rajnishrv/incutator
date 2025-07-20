import cv2
import mediapipe as mp
import json
import time
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# File to share pose data with Blender
POSE_DATA_FILE = "pose_data.json"

# Bone mapping - adjust these to match your Blender armature
BONE_MAPPING = {
    0: "head",           # Nose
    11: "left_shoulder", # Left shoulder
    12: "right_shoulder",# Right shoulder
    13: "left_elbow",    # Left elbow
    14: "right_elbow",   # Right elbow
    15: "left_wrist",    # Left wrist
    16: "right_wrist",   # Right wrist
    23: "left_hip",      # Left hip
    24: "right_hip",     # Right hip
    25: "left_knee",     # Left knee
    26: "right_knee",    # Right knee
    27: "left_ankle",    # Left ankle
    28: "right_ankle"    # Right ankle
}

cap = cv2.VideoCapture(0)  # Open webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# For FPS calculation
prev_time = 0

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for selfie view and convert color
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Get body pose landmarks
        results = pose.process(image_rgb)
        
        # Prepare pose data for Blender
        pose_data = {}
        
        if results.pose_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            
            # Process each landmark
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in BONE_MAPPING:
                    # Convert to Blender coordinate system
                    pose_data[BONE_MAPPING[idx]] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z if results.pose_world_landmarks else 0.5,
                        "visibility": landmark.visibility
                    }
            
            # Write pose data to file
            with open(POSE_DATA_FILE, 'w') as f:
                json.dump(pose_data, f)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Pose Detection (Press ESC to quit)', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()