import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import initialize_video_capture, initialize_video_writer
from detectionhandler import process_player_detections, process_ball_detection, optical_flow_tracking, calculate_iou
from drawhelper import draw_transparent_ellipse, draw_inverted_triangle
from events import detect_events, TRAJECTORY_LENGTH
import os
from collections import deque
from coordinateconverter import Detection, CoordinateConverter
from filepaths import FILE_PATH


def get_dominant_color(image, bbox):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 255, 255, 255  # Return white as a default color
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return 255, 255, 255  # Return white as a default color
    roi = image[y:y + h, x:x + w]
    average_color = np.nanmean(roi, axis=(0, 1))
    if np.isnan(average_color).any():
        return 255, 255, 255  # Return white as a default color
    max_channel = max(average_color)
    adjusted_color = tuple(min(int(value * 2) if value == max_channel else int(value), 255) for value in average_color)
    return adjusted_color


def main(video_path, output_path='output.mp4'):
    model = YOLO('yolov8m.pt', verbose=False)
    tracker = DeepSort(max_age=15, nms_max_overlap=0.7, nn_budget=100, max_iou_distance=0.7)
    cap, fps, frame_width, frame_height = initialize_video_capture(video_path)
    out = initialize_video_writer(output_path, fps, frame_width, frame_height)

    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (frame_width, frame_height))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    prev_ball = None
    ball_positions = deque(maxlen=TRAJECTORY_LENGTH)  # Store the last N ball positions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, verbose=False)
        detections = process_player_detections(results)
        ball_bbox = process_ball_detection(results)

        # Use optical flow to track the ball if it was not detected
        if ball_bbox is None and prev_ball is not None:
            ball_bbox = optical_flow_tracking(old_gray, frame_gray, prev_ball)

        if ball_bbox is not None:
            # Store the ball's center position
            # ball_center = (int(ball_bbox.x + ball_bbox.width / 2), int(ball_bbox.y + ball_bbox.height / 2))
            ball_positions.append((ball_bbox.center.x, ball_bbox.center.y))

            # Extract player positions from detections
            player_positions = [
                (int(d.x + d.width / 2), int(d.y + d.height / 2))  # Use center of bounding box
                for d in detections if d.class_id == 0  # Only include players
            ]

            # Detect events (shot, pass, or kick)
            detect_events(ball_positions, player_positions, frame)

            draw_transparent_ellipse(frame,
                                     ball_bbox.bottom_center,
                                     (ball_bbox.width // 2,
                                      max(ball_bbox.height // 6, 5)),
                                     (0, 0, 0),
                                     1
                                     )
            draw_inverted_triangle(frame, ball_bbox.top_center, 6)

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

        prev_ball = ball_bbox
        old_gray = frame_gray.copy()
        out.write(frame)
        cv2.imshow('Football Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    grandparent_dir = os.path.dirname(os.path.dirname(__file__))
    main(os.path.join(grandparent_dir, FILE_PATH))
