import cv2
import numpy as np
import json

import config
from utils.image_processing import get_object_bounding_box, segment_target_with_sam, detect_objects_in_mask_region, find_table_surface_from_mask
from logger_config import logger
from api.stability_ai_api import run_stability_ai_blending_approach

def run_logic_only_approach(use_sam=False):
    """
    Fits a container into a rack using a scaled, perspective-correct geometric warp
    based on the object's bounding box and alpha channel from the container PNG.
    """
    if use_sam:
        logger.info("--- Running Logic (Geometric) Approach with SAM ---")
    else:
        logger.info("--- Running Logic (Geometric) Approach with JSON ---")

    try:
        rack_img = cv2.imread(config.BASE_IMAGE_PATH)
        (x, y, w, h), container_img_4channel = get_object_bounding_box(config.OVERLAY_IMAGE_PATH)

        if rack_img is None or container_img_4channel is None:
            logger.error("Could not load one or more image files.")
            return

        if use_sam:
            with open(config.MEASUREMENTS_PATH, 'r') as f:
                measurements = json.load(f)
            prompt_point = measurements['base']['prompt_point']
            # Get the target mask from SAM
            target_mask = segment_target_with_sam(config.BASE_IMAGE_PATH, prompt_point)
            if target_mask is None:
                logger.error("Could not segment the target object.")
                return

            # Find the contours of the target mask
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.error("No contours found in the target mask.")
                return

            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x_target, y_target, w_target, h_target = cv2.boundingRect(largest_contour)

            # Calculate overlay dimensions for surface detection
            table_center_x = x_target + (w_target // 2)

            # Use a more reasonable overlay width (scale down from table width)
            # Most overlays will be smaller than the full table width
            estimated_overlay_width = min(w_target // 2, 300)  # Cap at 300px, use half table width

            # Find the table surface from the mask, focusing on center area
            surface_info = find_table_surface_from_mask(target_mask, estimated_overlay_width, table_center_x)

            if surface_info['surface_y'] is not None:
                # Use table surface for placement
                table_surface_y = surface_info['surface_y']
                logger.info(f"Using table surface at Y={table_surface_y} for overlay placement")

                # Create placement area above the table surface
                # The overlay should sit ON TOP of the table, not in front of it
                placement_height = h_target // 2  # Use half the table height for overlay area
                placement_y = table_surface_y - placement_height  # Place above the surface

                full_dst_quad = np.array([
                    [x_target, placement_y],
                    [x_target + w_target, placement_y],
                    [x_target + w_target, table_surface_y],  # Bottom touches table surface
                    [x_target, table_surface_y]
                ], dtype="float32")
            else:
                # Fallback to original bounding box method
                logger.warning("Could not find table surface, using bounding box method")
                full_dst_quad = np.array([
                    [x_target, y_target],
                    [x_target + w_target, y_target],
                    [x_target + w_target, y_target + h_target],
                    [x_target, y_target + h_target]
                ], dtype="float32")
        else:
            with open(config.MEASUREMENTS_PATH, 'r') as f:
                measurements = json.load(f)
            tl = measurements['base']['position']['top_left']
            br = measurements['base']['position']['bottom_right']
            tr = [br[0], tl[1]]
            bl = [tl[0], br[1]]
            full_dst_quad = np.array([tl, tr, br, bl], dtype="float32")


    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Make sure all required files are in the directory.")
        return
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

    logger.info(f"Detected object at [x={x}, y={y}, w={w}, h={h}] in the container image.")
    src_quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")

    # Check for existing objects in the target region before placement
    if use_sam:
        # Use the area above the table surface for object detection
        if surface_info['surface_y'] is not None:
            table_surface_y = surface_info['surface_y']
            placement_height = h_target // 2
            placement_y = table_surface_y - placement_height
            target_region = [x_target, placement_y, w_target, placement_height]
        else:
            target_region = [x_target, y_target, w_target, h_target]
    else:
        with open(config.MEASUREMENTS_PATH, 'r') as f:
            measurements = json.load(f)
        tl = measurements['base']['position']['top_left']
        br = measurements['base']['position']['bottom_right']
        target_region = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]

    detection_result = detect_objects_in_mask_region(target_mask, target_region)

    if detection_result['has_objects']:
        logger.warning("OBJECTS DETECTED in target placement area!")
        logger.warning(f"   Confidence: {detection_result['confidence']:.2f}")
        logger.warning(f"   Details: {detection_result['details']}")
        logger.warning("   Proceeding with overlay placement - objects may be occluded.")
    else:
        logger.info("No significant objects detected in target area. Safe to place overlay.")

    if use_sam:
        # Read measurements for reference, but calculate actual ratios from detected dimensions
        with open(config.MEASUREMENTS_PATH, 'r') as f:
            measurements = json.load(f)
        json_base_dims = measurements['base']
        json_overlay_dims = measurements['overlay']

        # Calculate the actual ratios from the JSON measurements
        json_ratio_w = json_overlay_dims['length'] / json_base_dims['length']
        json_ratio_h = json_overlay_dims['height'] / json_base_dims['height']

        # Use the JSON ratios to calculate the intended overlay size relative to the detected mask
        intended_overlay_w = w_target * json_ratio_w
        intended_overlay_h = h_target * json_ratio_h

        logger.info(f"Intended overlay size: {intended_overlay_w:.1f}x{intended_overlay_h:.1f}")

        # Set dimensions for the scaling calculation
        overlay_dims = {'length': intended_overlay_w, 'height': intended_overlay_h}
        base_dims = {'length': w_target, 'height': h_target}  # SAM detected mask area
    else:
        with open(config.MEASUREMENTS_PATH, 'r') as f:
            measurements = json.load(f)
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

    if use_sam and surface_info['surface_y'] is not None:
        # For SAM with table surface: position overlay with correct height, bottom touching surface
        table_surface_y = surface_info['surface_y']

        # Use the intended overlay dimensions (already calculated with ratios)
        intended_width = overlay_dims['length']
        intended_height = overlay_dims['height']

        # Calculate the center X position from the scaled quad
        center_x = (scaled_dst_quad[0][0] + scaled_dst_quad[1][0]) / 2

        # Position the overlay with correct dimensions
        final_dst_quad = np.array([
            [center_x - intended_width/2, table_surface_y - intended_height],  # Top-left
            [center_x + intended_width/2, table_surface_y - intended_height],  # Top-right
            [center_x + intended_width/2, table_surface_y],                    # Bottom-right
            [center_x - intended_width/2, table_surface_y]                     # Bottom-left
        ], dtype="float32")

        # Calculate final dimensions for verification
        final_width = int(np.linalg.norm(final_dst_quad[1] - final_dst_quad[0]))
        final_height = int(np.linalg.norm(final_dst_quad[3] - final_dst_quad[0]))

        logger.info(f"Positioning overlay at table surface (Y={table_surface_y}). Final dimensions: {final_width}x{final_height} pixels")
    else:
        # Original bottom alignment logic for JSON method or SAM fallback
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


def run_logic_approach(use_sam=False):
    """
    Fits a container into a rack using a scaled, perspective-correct geometric warp
    based on the object's bounding box and alpha channel from the container PNG.
    """
    run_logic_only_approach(use_sam=use_sam)
    # Run Stability AI blending for refinement
    run_stability_ai_blending_approach()
