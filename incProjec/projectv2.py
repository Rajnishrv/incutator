import cv2
import numpy as np

# Load video background
video = cv2.VideoCapture("VID_20240116_153942.mp4")

# Use webcam to detect ball
cap = cv2.VideoCapture(0)

# Tic-Tac-Toe board
rows, cols = 3, 3
game_board = [["" for _ in range(cols)] for _ in range(rows)]
current_player = "X"

# Track previous hit to avoid double marking
last_hit_cell = (-1, -1)

# Detect red ball using HSV color filtering
def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 100])
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

# Draw 3x3 grid and player marks
def draw_grid_and_marks(frame, cell_w, cell_h):
    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_h), (frame.shape[1], i * cell_h), (255, 255, 255), 2)
    for j in range(1, cols):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, frame.shape[0]), (255, 255, 255), 2)

    for i in range(rows):
        for j in range(cols):
            cx = j * cell_w + cell_w // 2
            cy = i * cell_h + cell_h // 2
            mark = game_board[i][j]
            if mark == "X":
                cv2.putText(frame, "X", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            elif mark == "O":
                cv2.putText(frame, "O", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

# Convert ball position to cell index
def get_cell_index(ball_pos, cell_w, cell_h):
    col = ball_pos[0] // cell_w
    row = ball_pos[1] // cell_h
    return int(row), int(col)

while True:
    ret_video, video_frame = video.read()
    ret_cam, cam_frame = cap.read()

    if not ret_video or not ret_cam:
        break

    cam_frame = cv2.flip(cam_frame, 1)

    # Resize video frame if needed
    height, width = cam_frame.shape[:2]
    video_frame = cv2.resize(video_frame, (width, height))

    # Calculate cell sizes
    cell_width = width // cols
    cell_height = height // rows

    # Detect ball
    ball = detect_ball(cam_frame)

    # Draw grid and marks
    draw_grid_and_marks(video_frame, cell_width, cell_height)

    # Detect hit and update game board
    if ball:
        cv2.circle(cam_frame, ball, 10, (255, 255, 255), -1)
        row, col = get_cell_index(ball, cell_width, cell_height)

        if 0 <= row < rows and 0 <= col < cols:
            if game_board[row][col] == "" and (row, col) != last_hit_cell:
                game_board[row][col] = current_player
                last_hit_cell = (row, col)
                current_player = "O" if current_player == "X" else "X"

    # Show frames
    cv2.imshow("Projected Video (Tic-Tac-Toe)", video_frame)
    cv2.imshow("Webcam Ball Detection", cam_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
