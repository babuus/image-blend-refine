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

def compare_images(image1_path, image2_path, output_path):
    """
    Compares two images and creates a heatmap of the differences.
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        logger.error("Could not load one or more image files for comparison.")
        return

    if img1.shape != img2.shape:
        # Resize the larger image to the size of the smaller image
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        if height1 * width1 > height2 * width2:
            img1 = cv2.resize(img1, (width2, height2))
        else:
            img2 = cv2.resize(img2, (width1, height1))

    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        custom_colormap[i, 0, 1] = 255 - i  # Green
        custom_colormap[i, 0, 2] = i      # Red

    heatmap = cv2.applyColorMap(diff_gray, custom_colormap)
    
    cv2.imwrite(output_path, heatmap)
    logger.info(f"Successfully created comparison heatmap: '{output_path}'")