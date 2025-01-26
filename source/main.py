import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import initialize_video_capture, initialize_video_writer
import os
from collections import Counter



def initialize_model():
    return YOLO('yolov8m.pt')


def initialize_tracker():
    return DeepSort(max_age=100, nms_max_overlap=0.8, nn_budget=150)


def process_detections(results):
    detections = []
    ball_bbox = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in [0, 32]:  # 0 = Person, 32 = Ball
                continue
            conf = float(box.conf[0])
            if conf < 0.2:  # Konfidenz niedriger setzen, um mehr Erkennungen zu bekommen
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            detections.append(([x1, y1, width, height], conf, class_id))

            # Speichere die Ballposition für Tracking
            if class_id == 32:
                ball_bbox = (x1, y1, x2, y2)

    return detections, ball_bbox


def optical_flow_tracking(old_gray, frame_gray, prev_ball):
    """Verfolge den Ball mit optischem Fluss, falls YOLO ihn nicht erkennt."""
    if prev_ball is None:
        return None

    x1, y1, x2, y2 = prev_ball
    center = np.array([[[(x1 + x2) // 2, (y1 + y2) // 2]]], dtype=np.float32)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, center, None, **lk_params)

    if st[0][0] == 1:  # Falls das Tracking erfolgreich war
        new_x, new_y = p1[0][0]
        return (
        int(new_x - (x2 - x1) / 2), int(new_y - (y2 - y1) / 2), int(new_x + (x2 - x1) / 2), int(new_y + (y2 - y1) / 2))

    return None  # Falls Tracking fehlschlägt


def draw_transparent_ellipse(frame, center, axes, color, thickness=2):
    overlay = frame.copy()
    cv2.ellipse(overlay,
                center=center,
                axes=axes,
                angle=0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_4
                )
    alpha = 0.5  # Transparenzgrad
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_inverted_triangle(frame, center, size):
    overlay = frame.copy()
    triangle_pts = np.array([
        [center[0] - size, center[1] - size],
        [center[0] + size, center[1] - size],
        [center[0], center[1] + size]
    ], np.int32)
    cv2.fillPoly(overlay, [triangle_pts], (0, 0, 0))
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def get_dominant_color(image, bbox):
    """Berechnet den Mittelwert der Farben im oberen Bereich der Bounding Box und verdoppelt den höchsten Wert."""
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    # Berechne den Mittelwert der Farben im ROI
    average_color = np.mean(roi, axis=(0, 1))

    # Identifiziere den höchsten Farbwert
    max_channel = max(average_color)

    # Verdopple den höchsten Wert, stelle aber sicher, dass er nicht über 255 geht
    adjusted_color = tuple(
        min(int(value * 2) if value == max_channel else int(value), 255) for value in average_color)

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

        if ball_bbox is None and prev_ball is not None:  # Falls YOLO den Ball nicht findet
            ball_bbox = optical_flow_tracking(old_gray, frame_gray, prev_ball)

        if ball_bbox is not None:
            x1, y1, x2, y2 = ball_bbox
            center = ((x1 + x2) // 2, y2)  # Am unteren Rand des Balls
            axes = (max(abs(x2 - x1) // 2, 10), max(abs(y2 - y1) // 6, 5))
            draw_transparent_ellipse(frame, center, axes, (0, 0, 0), 1)
            draw_inverted_triangle(frame, ((x1 + x2) // 2, y1 - 15), 6)
            prev_ball = ball_bbox

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.get_det_class() == 32:
                continue
            track_id = track.track_id
            cx, cy, width, height = track.to_tlwh()
            center = (int(cx + width // 2), int(cy + height))
            axes = (int(width), int(0.35 * width))

            dominant_color = get_dominant_color(frame, (int(cx), int(cy), int(width), int(height)))
            draw_transparent_ellipse(frame, center, axes, dominant_color, 2)

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
    main(os.path.join(grandparent_dir, "examples/TuerkGuecu_1.mp4"))
