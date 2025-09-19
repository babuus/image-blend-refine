import cv2
import numpy as np
import json

import config
from utils.image_processing import get_object_bounding_box
from logger_config import logger
from api.stability_ai_api import run_stability_ai_blending_approach

def run_logic_only_approach():
    """
    Fits a container into a rack using a scaled, perspective-correct geometric warp
    based on the object's bounding box and alpha channel from the container PNG.
    """
    logger.info("--- Running Logic (Geometric) Approach ---")
    try:
        rack_img = cv2.imread(config.BASE_IMAGE_PATH)
        (x, y, w, h), container_img_4channel = get_object_bounding_box(config.OVERLAY_IMAGE_PATH)
        with open(config.MEASUREMENTS_PATH, 'r') as f:
            measurements = json.load(f)

        if rack_img is None or container_img_4channel is None:
            logger.error("Could not load one or more image files.")
            return

    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Make sure all required files are in the directory.")
        return

    logger.info(f"Detected object at [x={x}, y={y}, w={w}, h={h}] in the container image.")
    src_quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")

    tl = measurements['base']['position']['top_left']
    br = measurements['base']['position']['bottom_right']
    tr = [br[0], tl[1]]
    bl = [tl[0], br[1]]
    full_dst_quad = np.array([tl, tr, br, bl], dtype="float32")

    base_dims = measurements['base']
    overlay_dims = measurements['overlay']

    if base_dims['length'] == 0 or base_dims['height'] == 0:
        logger.error("Base dimensions cannot be zero.")
        return

    width_ratio = overlay_dims['length'] / base_dims['length']
    height_ratio = overlay_dims['height'] / base_dims['height']
    logger.info(f"Overlay to base size ratios -> Width: {width_ratio:.2f}, Height: {height_ratio:.2f}")

    center_point = np.mean(full_dst_quad, axis=0)
    scaled_dst_quad = np.zeros_like(full_dst_quad)
    for i, point in enumerate(full_dst_quad):
        vec = point - center_point
        vec[0] *= width_ratio
        vec[1] *= height_ratio
        scaled_dst_quad[i] = center_point + vec

    full_bottom_y = (full_dst_quad[2][1] + full_dst_quad[3][1]) / 2
    scaled_bottom_y = (scaled_dst_quad[2][1] + scaled_dst_quad[3][1]) / 2
    y_shift = full_bottom_y - scaled_bottom_y
    
    final_dst_quad = scaled_dst_quad.copy()
    final_dst_quad[:, 1] += y_shift
    logger.info(f"Bottom-aligning overlay by shifting {y_shift:.2f} pixels down.")

    logger.info("Calculating perspective...")
    M = cv2.getPerspectiveTransform(src_quad, final_dst_quad)
    warped_container_4channel = cv2.warpPerspective(container_img_4channel, M, (rack_img.shape[1], rack_img.shape[0]))

    logger.info("Pasting overlay into base using alpha channel...")
    warped_bgr = warped_container_4channel[:, :, :3]
    warped_alpha = warped_container_4channel[:, :, 3]
    inv_alpha = cv2.bitwise_not(warped_alpha)
    rack_bg = cv2.bitwise_and(rack_img, rack_img, mask=inv_alpha)
    container_fg = cv2.bitwise_and(warped_bgr, warped_bgr, mask=warped_alpha)
    output_img = cv2.add(rack_bg, container_fg)

    cv2.imwrite(config.LOGIC_OUTPUT_PATH, output_img)
    logger.info(f"Successfully created '{config.LOGIC_OUTPUT_PATH}'")

def run_logic_approach():
    """
    Fits a container into a rack using a scaled, perspective-correct geometric warp
    based on the object's bounding box and alpha channel from the container PNG.
    """
    run_logic_only_approach()
    # Run Stability AI blending for refinement
    run_stability_ai_blending_approach()
