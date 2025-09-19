import cv2
import numpy as np
import base64
from logger_config import logger

def get_object_bounding_box(image_path):
    """Finds the tightest bounding box around the non-transparent object in a PNG."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        return None, None

    if img.shape[2] < 4:
        h, w = img.shape[:2]
        return (0, 0, w, h), img

    alpha_channel = img[:, :, 3]
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning(f"No object found in {image_path}, returning full image dimensions.")
        h, w = img.shape[:2]
        return (0, 0, w, h), img

    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour), img


def get_file_content_as_base64(path):
    """Reads a file and returns its base64 encoded content."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None