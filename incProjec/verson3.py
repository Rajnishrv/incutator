import cv2
import numpy as np

# Video background (optional)
video = cv2.VideoCapture("VID_20240116_153942.mp4")  # Or use a blank background

# Webcam to track ball
cap = cv2.VideoCapture(0)

# Tic-Tac-Toe setup
rows, cols = 3, 3
game_board = [["" for _ in range(cols)] for _ in range(rows)]
current_player = "X"
last_hit = (-1, -1)

# Detect red ball
def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            return int(x), int(y)
    return None

# Draw the 3x3 grid and marks
def draw_grid_and_marks(frame, cell_w, cell_h):
    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_h), (frame.shape[1], i * cell_h), (255, 255, 255), 2)
    for j in range(1, cols):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, frame.shape[0]), (255, 255, 255), 2)

    for i in range(rows):
        for j in range(cols):
            mark = game_board[i][j]
            if mark != "":
                cx = j * cell_w + cell_w // 2
                cy = i * cell_h + cell_h // 2
                color = (0, 255, 0) if mark == "X" else (0, 0, 255)
                cv2.putText(frame, mark, (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

# Determine which grid cell was hit
def get_hit_cell(ball_x, ball_y, cell_w, cell_h):
    col = ball_x // cell_w
    row = ball_y // cell_h
    return int(row), int(col)

# Optional: Check if someone won
def check_winner():
    for i in range(3):
        if game_board[i][0] == game_board[i][1] == game_board[i][2] != "":
            return game_board[i][0]
        if game_board[0][i] == game_board[1][i] == game_board[2][i] != "":
            return game_board[0][i]
    if game_board[0][0] == game_board[1][1] == game_board[2][2] != "":
        return game_board[0][0]
    if game_board[0][2] == game_board[1][1] == game_board[2][0] != "":
        return game_board[0][2]
    return None

while True:
    ret_video, video_frame = video.read()
    ret_cam, cam_frame = cap.read()

    if not ret_video or not ret_cam:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    height, width = cam_frame.shape[:2]
    video_frame = cv2.resize(video_frame, (width, height))

    cell_w = width // cols
    cell_h = height // rows

    # Draw grid and marks on projector video
    draw_grid_and_marks(video_frame, cell_w, cell_h)

    # Detect ball from webcam
    ball = detect_ball(cam_frame)
    if ball:
        x, y = ball
        cv2.circle(cam_frame, (x, y), 10, (255, 255, 255), -1)

        # Map ball position to grid cell on projector
        row, col = get_hit_cell(x, y, cell_w, cell_h)

        if 0 <= row < rows and 0 <= col < cols:
            if game_board[row][col] == "" and (row, col) != last_hit:
                game_board[row][col] = current_player
                last_hit = (row, col)
                current_player = "O" if current_player == "X" else "X"

    # Check for winner
    winner = check_winner()
    if winner:
        cv2.putText(video_frame, f"{winner} Wins!", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # Show game screen on projector
    cv2.imshow("Game - Move this window to projector", video_frame)

    # Show webcam feed for debugging
    cv2.imshow("Ball Detection", cam_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
