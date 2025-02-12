import cv2
from coordinateconverter import CoordinateConverter
import numpy as np


def draw_transparent_ellipse(frame,
                             center: CoordinateConverter,
                             axes: tuple[float, float],
                             color: tuple[int, int, int],
                             thickness: int=2
                             ):
    """
    Draw an ellipse on the frame with the given parameters.

    Args:
        frame (numpy array): The frame to draw on
        center (CoordinateConverter): The center of the ellipse
        axes (tuple of floats): The length of the semi-axes of the ellipse
        color (tuple of ints): The color of the ellipse
        thickness (int): The thickness of the ellipse

    Returns:
        None
    """
    overlay = frame.copy()
    center_tuple = (int(center.x), int(center.y))
    axes_tuple = (int(axes[0]), int(axes[1]))
    cv2.ellipse(overlay, center_tuple, axes_tuple, 0, -45, 235, color, thickness, lineType=cv2.LINE_4)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


def draw_inverted_triangle(frame, center: CoordinateConverter, size: float) -> None:
    """
    Draw an inverted triangle on the frame with the given parameters.

    Args:
        frame (numpy array): The frame to draw on
        center (CoordinateConverter): The center of the triangle
        size (float): The size of the triangle

    Returns:
        None
    """
    overlay = frame.copy()
    triangle_pts = np.array([
        [center.x - size, center.y - size],
        [center.x + size, center.y - size],
        [center.x, center.y + size]
    ], np.int32)
    cv2.fillPoly(overlay, [triangle_pts], (0, 0, 0))
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)