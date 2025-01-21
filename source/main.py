import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def initialize_model():
    return YOLO('yolov8m.pt')


def initialize_tracker():
    return DeepSort(max_age=70, nms_max_overlap=1.0, nn_budget=100)


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height


def initialize_video_writer(output_path, fps, width, height):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def process_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in [0, 32]:
                continue

            conf = float(box.conf[0])
            if conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2, y2], conf, class_id))
    return detections


def draw_triangle(frame, x1, y1, x2, y2, color):
    points = np.array([
        [x1, int(y1 - 0.2 * (y2 - y1)) - 15],
        [x2, int(y1 - 0.2 * (y2 - y1)) - 15],
        [(x1 + x2) // 2, y1 - 15]
    ])
    cv2.drawContours(frame, [points], 0, color, -1)


def annotate_frame(frame, tracks, detections):
    for track, (bbox, _, class_id) in zip(tracks, detections):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = bbox
        color = (0, 0, 0) if class_id == 0 else (0, 255, 0)

        draw_triangle(frame, x1, y1, x2, y2, color)
        text_position = (x1 - 20, y1) if class_id == 0 else (x1 - 35, y1)
        cv2.putText(frame, f'ID {track_id}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main(video_path, output_path='output.mp4'):
    model = initialize_model()
    tracker = initialize_tracker()
    cap, fps, width, height = initialize_video_capture(video_path)
    out = initialize_video_writer(output_path, fps, width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detections = process_detections(results)
        tracks = tracker.update_tracks(detections, frame=frame)

        annotate_frame(frame, tracks, detections)
        out.write(frame)
        cv2.imshow('Football Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\Lenovo\OneDrive\Dokumente\MatchMetrics\examples\TuerkGuecu_1.mp4"
    main(video_path)
