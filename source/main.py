import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os


def initialize_model():
    return YOLO('yolov8m.pt')


def initialize_tracker():
    return DeepSort(max_age=100, nms_max_overlap=0.8, nn_budget=150)


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 1.25)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 1.25)
    return cap, fps, width, height


def initialize_video_writer(output_path, fps, width, height):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


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
            detections.append(([cx, cy, width, height], conf, class_id))

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


def main(video_path, output_path='output.mp4'):
    model = initialize_model()
    tracker = initialize_tracker()
    cap, fps, frame_width, frame_height = initialize_video_capture(video_path)
    out = initialize_video_writer(output_path, fps, frame_width, frame_height)

    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (frame_width, frame_height))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    prev_ball = None  # Letzte bekannte Position des Balls

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Ball in Blau markieren
            prev_ball = ball_bbox  # Speichere die neue Ballposition

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            cx, cy, width, height = track.to_tlwh()
            color = (0, 255, 0) if track.get_det_class() == 0 else (255, 0, 0)
            cv2.putText(frame, f'ID{track_id}', (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
