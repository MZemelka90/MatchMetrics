import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import initialize_video_capture, initialize_video_writer
import os
from collections import Counter
from detection import Detection, CoordinateConverter

def initialize_model():
    return YOLO('yolov8m.pt')

def initialize_tracker():
    # Tune DeepSort parameters for better tracking
    return DeepSort(max_age=15, nms_max_overlap=0.7, nn_budget=100, max_iou_distance=0.7)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def process_detections(results):
    detections = []
    ball_bbox = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in [0, 32]:  # 0: person, 32: sports ball
                continue
            conf = float(box.conf[0])
            if conf < 0.5:  # Increased confidence threshold for detections
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(x1, y1, x2 - x1, y2 - y1, conf, class_id))

            if class_id == 32:
                ball_bbox = Detection(x1, y1, x2 - x1, y2 - y1, conf, class_id)

    return detections, ball_bbox

def optical_flow_tracking(old_gray, frame_gray, prev_ball: Detection):
    if prev_ball is None:
        return None

    x1, y1, x2, y2 = prev_ball.x, prev_ball.y, prev_ball.x + prev_ball.width, prev_ball.y + prev_ball.height
    center = np.array([[[(x1 + x2) // 2, (y1 + y2) // 2]]], dtype=np.float32)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, center, None, **lk_params)

    if st[0][0] == 1:
        new_x, new_y = p1[0][0]
        return Detection(new_x - (x2 - x1) / 2, new_y - (y2 - y1) / 2, x2 - x1, y2 - y1, prev_ball.conf, prev_ball.class_id)

    return None

def draw_transparent_ellipse(frame, center: CoordinateConverter, axes, color, thickness=2):
    overlay = frame.copy()
    center_tuple = (int(center.x), int(center.y))
    axes_tuple = (int(axes[0]), int(axes[1]))
    cv2.ellipse(overlay, center_tuple, axes_tuple, 0, -45, 235, color, thickness, lineType=cv2.LINE_4)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def draw_inverted_triangle(frame, center: CoordinateConverter, size):
    overlay = frame.copy()
    triangle_pts = np.array([
        [center.x - size, center.y - size],
        [center.x + size, center.y - size],
        [center.x, center.y + size]
    ], np.int32)
    cv2.fillPoly(overlay, [triangle_pts], (0, 0, 0))
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def get_dominant_color(image, bbox):
    """
    Calculate the dominant color in the bounding box region.
    Returns a default color (white) if the bounding box is invalid.
    """
    x, y, w, h = bbox

    # Check if the bounding box is valid
    if w <= 0 or h <= 0:
        return (255, 255, 255)  # Return white as a default color

    # Ensure the bounding box is within the image boundaries
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # If the bounding box is still invalid, return a default color
    if w <= 0 or h <= 0:
        return (255, 255, 255)  # Return white as a default color

    # Extract the region of interest (ROI)
    roi = image[y:y + h, x:x + w]

    # Calculate the average color in the ROI
    average_color = np.nanmean(roi, axis=(0, 1))  # Use nanmean to ignore NaN values

    # If average_color contains NaN values, return a default color
    if np.isnan(average_color).any():
        return (255, 255, 255)  # Return white as a default color

    # Identify the dominant channel and adjust the color
    max_channel = max(average_color)
    adjusted_color = tuple(min(int(value * 2) if value == max_channel else int(value), 255) for value in average_color)

    return adjusted_color

def main(video_path, output_path='output.mp4'):
    model = initialize_model()
    tracker = initialize_tracker()
    cap, fps, frame_width, frame_height = initialize_video_capture(video_path)
    out = initialize_video_writer(output_path, fps, frame_width, frame_height)

    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (frame_width, frame_height))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    prev_ball = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, verbose=False)
        detections, ball_bbox = process_detections(results)

        if ball_bbox is None and prev_ball is not None:
            ball_bbox = optical_flow_tracking(old_gray, frame_gray, prev_ball)

        if ball_bbox is not None:
            draw_transparent_ellipse(frame,
                                     ball_bbox.bottom_center,
                                     (ball_bbox.width // 2,
                                      max(ball_bbox.height // 6, 5)),
                                     (0, 0, 0),
                                     1
                                     )
            draw_inverted_triangle(frame, ball_bbox.top_center, 6)
            prev_ball = ball_bbox

        # Update tracks with detections
        tracks = tracker.update_tracks(
            [([d.x, d.y, d.width, d.height], d.conf, d.class_id) for d in detections],
            frame=frame
        )

        # Draw tracks only if they are confirmed and not the ball
        for track in tracks:
            if not track.is_confirmed() or track.get_det_class() == 32:
                continue
            track_id = track.track_id
            cx, cy, width, height = track.to_tlwh()
            track_det = Detection(int(cx), int(cy), int(width), int(height), 1, track_id)

            # Calculate feet position (bottom center of the bounding box)
            feet_position = CoordinateConverter(int(cx + width // 2), int(cy + height))

            # Custom overlap check to prevent duplicate tracks
            for detection in detections:
                if detection.class_id == 0:  # Only check for players
                    iou = calculate_iou(
                        [cx, cy, cx + width, cy + height],
                        [detection.x, detection.y, detection.x + detection.width, detection.y + detection.height]
                    )
                    if iou > 0.5:  # If overlap is significant, associate detection with existing track
                        break
            else:
                # If no significant overlap, skip drawing this track
                continue

            # Get dominant color and draw ellipse at feet position
            dominant_color = get_dominant_color(frame, (int(cx), int(cy), int(width), int(height)))
            draw_transparent_ellipse(frame, feet_position, (track_det.width // 2, track_det.height // 6), dominant_color, 2)

        old_gray = frame_gray.copy()
        out.write(frame)
        cv2.imshow('Football Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    grandparent_dir = os.path.dirname(os.path.dirname(__file__))
    main(os.path.join(grandparent_dir, "examples/TuerkGuecu_1.mp4"))