from coordinateconverter import Detection
import numpy as np
import cv2

def process_player_detections(results) -> list[Detection]:
    """Process YOLO results to extract player bounding box detections.

    Args:
        results: List of YOLO results, each containing a list of bounding boxes.

    Returns:
        list[Detection]: List of player bounding box detections.
    """
    detections = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            if class_id != 0:  # 0: person, 32: sports ball
                continue

            conf = float(box.conf[0])
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    conf=conf,
                    class_id=class_id
                )
            )
    return detections

def process_ball_detection(results) -> Detection|None:
    """
    Process YOLO results to extract ball bounding box detection.

    Args:
        results: List of YOLO results, each containing a list of bounding boxes.

    Returns:
        Detection|None: Ball bounding box detection, or None if no detection was found.
    """
    detection = None
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            if class_id != 32:  # 0: person, 32: sports ball
                continue

            conf = float(box.conf[0])
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection = Detection(x=x1, y=y1, width=x2 - x1, height=y2 - y1, conf=conf, class_id=class_id)
    return detection

def optical_flow_tracking(old_gray: np.ndarray, frame_gray: np.ndarray, prev_ball: Detection) -> Detection | None:
    """
    Tracks the ball's position using the Lucas-Kanade method with enhancements.

    Args:
    - old_gray (np.ndarray): Previous grayscale frame.
    - frame_gray (np.ndarray): Current grayscale frame.
    - prev_ball (Detection): The previous detection of the ball.

    Returns:
    - Detection | None: Updated ball detection if tracking is successful, otherwise None.
    """
    if prev_ball is None:
        return None

    # Define bounding box
    x1, y1, x2, y2 = prev_ball.x, prev_ball.y, prev_ball.x + prev_ball.width, prev_ball.y + prev_ball.height
    width, height = x2 - x1, y2 - y1

    # Track multiple key points (center + corners)
    points = np.array([
        [prev_ball.center.x, prev_ball.center.y],  # Center
        [x1, y1], [x2, y1],  # Top-left, Top-right
        [x1, y2], [x2, y2]  # Bottom-left, Bottom-right
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Optical Flow Parameters
    lk_params = dict(winSize=(max(15, int(width // 2)), max(15, int(height // 2))), # Dynamic window size
                     maxLevel=3,  # More pyramid levels for better accuracy
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02))

    # Compute new positions
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)

    # Filter valid points
    valid_points = p1[st == 1].reshape(-1, 2)

    if len(valid_points) == 0:
        return None  # Lost tracking

    # Compute new bounding box based on the tracked points
    new_x, new_y = np.mean(valid_points, axis=0)
    new_x -= width / 2
    new_y -= height / 2

    return Detection(new_x, new_y, width, height, prev_ball.conf, prev_ball.class_id)

def calculate_iou(box1: list, box2: list) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
    - box1, box2: (x1, y1, x2, y2) format

    Returns:
    - IoU value (float, 0 to 1)
    """
    # Unpack coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Compute intersection rectangle
    x_int_min = max(x1_min, x2_min)
    y_int_min = max(y1_min, y2_min)
    x_int_max = min(x1_max, x2_max)
    y_int_max = min(y1_max, y2_max)

    # Compute intersection area
    int_width = max(0, x_int_max - x_int_min)
    int_height = max(0, y_int_max - y_int_min)
    intersection_area = int_width * int_height

    # Compute areas
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU, ensuring division stability
    return intersection_area / max(union_area, 1e-9)