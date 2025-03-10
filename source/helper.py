import cv2

def initialize_video_capture(video_path: str, scale: float = 1.25):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // scale)
    return cap, fps, width, height


def initialize_video_writer(output_path: str, fps: int, width: int, height: int):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))