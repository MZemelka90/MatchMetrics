import numpy as np
from sklearn.linear_model import RANSACRegressor
import cv2

# Constants for event detection
SHOT_SPEED_THRESHOLD = 15  # Minimum horizontal speed for a shot (pixels/frame)
SHOT_CURVATURE_THRESHOLD = 0.005  # Maximum curvature for a shot
PASS_SPEED_THRESHOLD = 5  # Minimum speed for a pass (pixels/frame)
PASS_DISTANCE_THRESHOLD = 30  # Minimum distance for a pass (pixels)
MIN_KICK_DISTANCE = 30  # Minimum distance the ball must travel to be considered a kick (pixels)
TRAJECTORY_LENGTH = 5  # Number of ball positions to analyze
PROXIMITY_THRESHOLD = 80  # Maximum distance to consider a player close to the ball (pixels)


def detect_events(ball_positions, player_positions, frame):
    """
    Detect shot and pass events using robust trajectory analysis.
    """
    if len(ball_positions) < TRAJECTORY_LENGTH:
        return

    # Convert deque to list for slicing
    ball_positions_list = list(ball_positions)

    # Use the last N ball positions for analysis
    trajectory = np.array(ball_positions_list[-TRAJECTORY_LENGTH:])  # shape (N, 2)

    # Compute velocities between frames (in pixels/frame)
    velocities = np.diff(trajectory, axis=0)
    horiz_speed = np.linalg.norm(velocities[:, 0])
    avg_horiz_speed = np.mean(horiz_speed)

    # Use RANSAC to fit a quadratic curve for the vertical (y) component as a function of horizontal (x)
    X = trajectory[:, 0].reshape(-1, 1)
    y = trajectory[:, 1]
    poly_features = np.hstack([X, X**2])
    try:
        ransac = RANSACRegressor(min_samples=3, residual_threshold=5)
        ransac.fit(poly_features, y)
        curvature = abs(ransac.estimator_.coef_[1])  # Quadratic coefficient
    except Exception as e:
        print(f"RANSAC quadratic fit failed: {e}")
        curvature = 0

    # Detect shot: High horizontal speed and shallow curvature
    # if avg_horiz_speed > SHOT_SPEED_THRESHOLD and curvature < SHOT_CURVATURE_THRESHOLD:
        # cv2.putText(frame, "SHOT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Detect pass: High speed and large distance between consecutive positions
    # if avg_horiz_speed > PASS_SPEED_THRESHOLD and np.linalg.norm(trajectory[-1] - trajectory[-2]) > PASS_DISTANCE_THRESHOLD:
        # cv2.putText(frame, "PASS!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Detect kick: Ball moves away from a player after being close
    if len(player_positions) > 0:
        # Find the closest player to the ball at the start of the trajectory
        initial_ball_position = trajectory[0]
        closest_player = min(player_positions, key=lambda p: np.linalg.norm(np.array(p) - initial_ball_position))
        distance_to_player = np.linalg.norm(np.array(closest_player) - initial_ball_position)

        # Check if the ball was close to the player initially
        if distance_to_player < PROXIMITY_THRESHOLD:
            # Check if the ball has moved a minimum distance away from the player
            final_distance = np.linalg.norm(np.array(closest_player) - trajectory[-1])
            if final_distance - distance_to_player > MIN_KICK_DISTANCE:
                cv2.putText(frame, "KICK!", (closest_player[0] + 10, closest_player[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)