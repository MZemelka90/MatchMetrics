import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os


def initialize_model() -> YOLO:
    return YOLO('yolov8m.pt')


def initialize_tracker() -> DeepSort:
    return DeepSort(max_age=100, nms_max_overlap=0.8, nn_budget=150)


def initialize_video_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    print("fps: ", fps, "width: ", width, "height: ", height)
    return cap, fps, width, height


def initialize_video_writer(output_path: str, fps: int, width: int, height: int):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def process_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in [0, 32]:
                continue
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            detections.append(([cx, cy, width, height], conf, class_id))
    return detections


def draw_triangle(frame, cx, cy, height, color, size=10):
    points = np.array([
        [cx - size, cy - height // 2 - size],
        [cx + size, cy - height // 2 - size],
        [cx, cy - height // 2]
    ], dtype=np.int32)
    cv2.drawContours(frame, [points], 0, color, -1)

def draw_index(frame, index, color, cx, cy, height, size=15) -> None:
    position = (int(cx - size), int(cy - height // 2 - size))  # Stelle sicher, dass die Koordinaten Integer sind
    color = tuple(map(int, color))  # Stelle sicher, dass die Farbwerte Integer sind
    cv2.putText(frame, f'ID{index}', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def annotate_frame(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        cx, cy, width, height = track.to_tlwh()
        color = (0, 0, 0) if track.get_det_class() == 0 else (0, 255, 0)
        draw_triangle(frame, cx, cy, height, color)
        draw_index(frame, track_id, color, cx, cy, height)


def main(video_path: str, output_path: str = 'output.mp4'):
    model = initialize_model()
    tracker = initialize_tracker()
    cap, fps, width, height = initialize_video_capture(video_path)
    out = initialize_video_writer(output_path, fps, width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height))
        if not ret:
            break
        results = model(frame, verbose=False)
        detections = process_detections(results)

        tracks = tracker.update_tracks(detections, frame=frame)

        annotate_frame(frame, tracks)
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
