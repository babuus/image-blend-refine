import cv2
import numpy as np
import base64
from logger_config import logger
import config
from segment_anything import sam_model_registry, SamPredictor
import torch

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

def detect_objects_in_mask_region(mask, region_coords):
    """
    Detects if there are existing objects in a specified region using the existing SAM mask.
    This is much more efficient than running SAM again.

    Args:
        mask: Existing SAM mask (binary image where black=object, white=background)
        region_coords: List of [x, y, w, h] defining the region to check

    Returns:
        dict: {
            'has_objects': bool,
            'object_count': int,
            'confidence': float,
            'details': str
        }
    """
    logger.info("Analyzing SAM mask for objects in target region")
    try:
        x, y, w, h = region_coords

        # Validate mask dimensions
        if mask is None:
            logger.error("Mask is None")
            return {'has_objects': False, 'object_count': 0, 'confidence': 0.0, 'details': 'Error: Mask is None'}

        mask_height, mask_width = mask.shape[:2]

        # Clamp coordinates to mask bounds to prevent segmentation fault
        x = max(0, min(x, mask_width - 1))
        y = max(0, min(y, mask_height - 1))
        x_end = max(x + 1, min(x + w, mask_width))
        y_end = max(y + 1, min(y + h, mask_height))

        # Recalculate actual dimensions after clamping
        actual_w = x_end - x
        actual_h = y_end - y

        logger.info(f"Region bounds: original=[{region_coords[0]}, {region_coords[1]}, {region_coords[2]}, {region_coords[3]}], "
                   f"clamped=[{x}, {y}, {actual_w}, {actual_h}], mask_size=[{mask_width}, {mask_height}]")

        # Extract the region of interest from the existing mask
        roi_mask = mask[y:y_end, x:x_end]

        # In SAM mask: black (0) = table/object, white (255) = background/gaps
        # We want to detect "objects" which would be gaps/spaces in the table area
        # So we look for white areas (background) within the placement region

        # Count white pixels (background/gaps) in the region
        background_pixels = np.sum(roi_mask == 255)
        total_pixels = actual_w * actual_h
        background_ratio = background_pixels / total_pixels if total_pixels > 0 else 0

        # Find connected components of background areas (gaps)
        # Invert the ROI so gaps become white for connected components analysis
        gap_mask = (roi_mask == 255).astype(np.uint8) * 255

        # Find connected components (gaps between table structure)
        num_labels, labeled_mask = cv2.connectedComponents(gap_mask)
        gap_count = num_labels - 1  # Subtract background label

        # Filter out very small gaps (noise)
        significant_gaps = 0
        for label in range(1, num_labels):
            gap_area = np.sum(labeled_mask == label)
            if gap_area > total_pixels * 0.01:  # Gaps larger than 1% of region
                significant_gaps += 1

        # Determine if there are significant objects/gaps
        # High background ratio = lots of gaps = potential objects
        has_objects = background_ratio > 0.3 or significant_gaps > 2
        confidence = min(1.0, background_ratio * 2 + significant_gaps * 0.2)

        details = f"Background ratio: {background_ratio:.3f}, Significant gaps: {significant_gaps}, Total area: {total_pixels}"

        logger.info(f"Object detection result: {has_objects}, confidence: {confidence:.2f}")

        return {
            'has_objects': has_objects,
            'object_count': significant_gaps,
            'confidence': confidence,
            'details': details
        }

    except Exception as e:
        logger.error(f"Error during mask-based object detection: {e}")
        return {'has_objects': False, 'object_count': 0, 'confidence': 0.0, 'details': f'Error: {e}'}


# Keep the original SAM-based function for backwards compatibility if needed
def detect_objects_with_sam(image_path, region_coords, model_path="models/sam_vit_h_4b8939.pth", model_type="vit_h"):
    """
    DEPRECATED: Use detect_objects_in_mask_region() instead for better performance.
    This function runs SAM again which is inefficient.
    """
    logger.warning("Using deprecated detect_objects_with_sam(). Consider using detect_objects_in_mask_region() instead.")

    # For now, just return no objects to avoid double SAM calls
    return {'has_objects': False, 'object_count': 0, 'confidence': 0.0, 'details': 'Deprecated function - object detection skipped'}


def segment_target_with_sam(image_path, prompt_point, model_path="models/sam_vit_h_4b8939.pth", model_type="vit_h"):
    """
    Segments an object from an image using the Segment Anything Model (SAM) and a prompt point.
    """
    logger.info("Running SAM for target segmentation")
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image at {image_path}")
            return None

        device = "cpu"
        logger.info(f"Using device: {device}")

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        predictor.set_image(image)

        input_point = np.array([prompt_point])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        if not masks.any():
            logger.warning("No masks generated by SAM.")
            return None

        # The mask is a boolean array, convert it to a binary image
        target_mask = (masks[0] * 255).astype(np.uint8)

        logger.info("Successfully segmented the target object using SAM with prompt point.")
        cv2.imwrite(config.SAM_OUTPUT_PATH, target_mask)
        logger.info(f"Successfully saved SAM mask to: '{config.SAM_OUTPUT_PATH}'")
        return target_mask

    except Exception as e:
        logger.error(f"Error during SAM processing: {e}")
        return None


def find_shelf_bottom_surface(roi_mask, roi_left, overlay_width, table_center_x):
    """
    Finds the bottom surface of a shelf opening where items can be placed.

    Args:
        roi_mask: ROI of the mask containing the shelf opening (white pixels)
        roi_left: Left coordinate of the ROI in the full image
        overlay_width: Width of overlay for surface detection
        table_center_x: Center X coordinate

    Returns:
        dict: Surface information
    """
    logger.info("Finding bottom surface of shelf opening")
    try:
        # Find white areas (shelf openings)
        white_areas = (roi_mask == 255).astype(np.uint8)

        # Find contours of white areas (openings)
        contours, _ = cv2.findContours(white_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No shelf openings found")
            return {'surface_y': None, 'surface_points': [], 'confidence': 0.0}

        # Find the largest opening (main shelf compartment)
        largest_opening = max(contours, key=cv2.contourArea)

        # Get bounding box of the opening
        x, y, w, h = cv2.boundingRect(largest_opening)

        # Adjust coordinates back to full image
        full_x = x + roi_left
        full_y = y

        # The bottom of the opening is where items sit
        bottom_surface_y = y + h  # Bottom edge of the opening

        logger.info(f"Found shelf opening at [{full_x}, {full_y}, {w}, {h}], bottom surface at Y={bottom_surface_y}")

        # Create surface points along the bottom edge
        surface_points = []
        for px in range(x, x + w, max(1, w // 10)):  # Sample points along width
            surface_points.append([px + roi_left, bottom_surface_y])

        # High confidence for shelf bottoms since they're clearly defined
        confidence = 0.9

        return {
            'surface_y': bottom_surface_y,
            'surface_points': surface_points,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"Error finding shelf bottom surface: {e}")
        return {'surface_y': None, 'surface_points': [], 'confidence': 0.0}


def find_table_surface_from_mask(mask, overlay_width, table_center_x):
    """
    Finds the placement surface from a SAM mask. For tables, this finds the top surface.
    For shelves/compartments, this finds the bottom of the opening where items sit.

    Args:
        mask: Binary mask where black (0) = structure, white (255) = opening/background
        overlay_width: Width of the overlay image to be placed
        table_center_x: X coordinate of the center

    Returns:
        dict: {
            'surface_y': int,  # Y coordinate of placement surface
            'surface_points': list,  # [(x, y)] points along the surface
            'confidence': float
        }
    """
    logger.info("Finding table surface in center area")
    try:
        # Calculate the region of interest (center area matching overlay width)
        roi_half_width = overlay_width // 2
        roi_left = max(0, table_center_x - roi_half_width)
        roi_right = min(mask.shape[1], table_center_x + roi_half_width)


        # Extract only the center region from the mask
        roi_mask = mask[:, roi_left:roi_right]

        # Analyze mask to determine if this is a shelf/compartment or table surface
        white_pixels = np.sum(roi_mask == 255)
        black_pixels = np.sum(roi_mask == 0)
        total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
        white_ratio = white_pixels / total_pixels

        logger.info(f"ROI analysis: {white_ratio:.2f} white ratio, shape: {roi_mask.shape}")

        if white_ratio > 0.1:  # Significant white area suggests this is a shelf opening/compartment
            logger.info("Detected shelf/compartment scenario - looking for bottom of opening")
            # For shelves: white areas are openings where we place items
            # Find the bottom edge of the white opening
            return find_shelf_bottom_surface(roi_mask, roi_left, overlay_width, table_center_x)
        else:
            logger.info("Detected table surface scenario - looking for top surface")
            # For tables: invert mask to find table structure surfaces
            inverted_roi_mask = cv2.bitwise_not(roi_mask)

        # Find contours of the table in the ROI
        contours, _ = cv2.findContours(inverted_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No table contours found in center ROI")
            return {'surface_y': None, 'surface_points': [], 'confidence': 0.0}

        logger.info(f"Found {len(contours)} contours in ROI. Mask shape: {mask.shape}, ROI: [{roi_left}:{roi_right}]")

        # Find the largest contour (main table in center area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Adjust contour points back to full image coordinates
        contour_points = largest_contour.reshape(-1, 2)
        contour_points[:, 0] += roi_left  # Adjust X coordinates

        # For shelf/rack detection, we need to find the bottom surface where items can be placed
        # Instead of just looking at topmost points, analyze the geometry
        min_y = np.min(contour_points[:, 1])
        max_y = np.max(contour_points[:, 1])
        tolerance = 15  # pixels tolerance for "same level"

        logger.info(f"Contour Y range: {min_y} to {max_y} (height: {max_y - min_y})")
        logger.info(f"Overlay width for detection: {overlay_width}, Table center X: {table_center_x}")

        # Look for the actual shelf surface - for shelves, this is often the bottom edge
        # Check multiple Y levels to find a substantial horizontal surface
        best_surface_y = None
        best_surface_points = []
        best_confidence = 0.0

        # For shelves/racks, check both top and bottom areas more thoroughly
        search_range = max(100, (max_y - min_y) // 2)  # Adaptive search range

        for y_offset in range(0, search_range, 3):  # Check Y levels from top down
            current_y = min_y + y_offset

            # Find points at this Y level
            surface_candidates = contour_points[
                (contour_points[:, 1] >= current_y - tolerance) &
                (contour_points[:, 1] <= current_y + tolerance)
            ]

            if len(surface_candidates) > 0:
                # Sort by X coordinate
                surface_points = surface_candidates[np.argsort(surface_candidates[:, 0])]

                # Check if this forms a reasonable horizontal surface
                x_span = surface_points[-1][0] - surface_points[0][0] if len(surface_points) > 1 else 0
                surface_width_ratio = x_span / overlay_width if overlay_width > 0 else 0

                # Good surface should span at least 30% of the overlay width (reduced threshold)
                if surface_width_ratio > 0.3:
                    confidence = min(1.0, surface_width_ratio + len(surface_points) * 0.01)

                    if confidence > best_confidence:
                        best_surface_y = int(current_y)
                        best_surface_points = surface_points
                        best_confidence = confidence

        # For shelf scenarios, also check the bottom area more thoroughly
        # This often represents the actual shelf surface
        bottom_start = min_y + (max_y - min_y) * 0.6  # Check bottom 40% of the detected area
        for y_offset in range(int(bottom_start - min_y), search_range, 3):
            current_y = min_y + y_offset

            surface_candidates = contour_points[
                (contour_points[:, 1] >= current_y - tolerance) &
                (contour_points[:, 1] <= current_y + tolerance)
            ]

            if len(surface_candidates) > 0:
                surface_points = surface_candidates[np.argsort(surface_candidates[:, 0])]
                x_span = surface_points[-1][0] - surface_points[0][0] if len(surface_points) > 1 else 0
                surface_width_ratio = x_span / overlay_width if overlay_width > 0 else 0

                # Give higher preference to bottom surfaces for shelf scenarios
                if surface_width_ratio > 0.2:  # Lower threshold for bottom surfaces
                    confidence = min(1.0, surface_width_ratio + len(surface_points) * 0.01 + 0.3)  # Bonus for bottom

                    if confidence > best_confidence:
                        best_surface_y = int(current_y)
                        best_surface_points = surface_points
                        best_confidence = confidence

        if best_surface_y is not None:
            logger.info(f"Table surface found at Y={best_surface_y}")
        else:
            logger.warning("No suitable horizontal surface found in center area")

        return {
            'surface_y': best_surface_y,
            'surface_points': best_surface_points.tolist() if len(best_surface_points) > 0 else [],
            'confidence': best_confidence
        }

    except Exception as e:
        logger.error(f"Error finding table surface: {e}")
        return {'surface_y': None, 'surface_points': [], 'confidence': 0.0}


def visualize_table_detection(base_image_path, mask_path, output_path):
    """
    Creates a visualization showing detected table areas from SAM mask.

    Args:
        base_image_path: Path to the original base image
        mask_path: Path to the SAM mask
        output_path: Path to save the visualization

    Returns:
        dict: Analysis results with table components info
    """
    logger.info("--- Creating table detection visualization ---")
    try:
        # Load base image and mask
        base_image = cv2.imread(base_image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if base_image is None or mask is None:
            logger.error("Could not load base image or mask")
            return None

        # Create visualization image
        vis_image = base_image.copy()

        # Invert mask: table areas become white (255), background becomes black (0)
        inverted_mask = cv2.bitwise_not(mask)

        # Find all contours (table components)
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze each contour
        table_components = []
        colors = [
            (0, 255, 0),    # Green for main table surface
            (255, 0, 0),    # Blue for table legs
            (0, 255, 255),  # Yellow for other components
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip very small areas (noise)
            if area < 500:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Analyze component type based on dimensions
            aspect_ratio = w / h if h > 0 else 0

            if aspect_ratio > 2:  # Wide = likely table surface
                component_type = "table_surface"
                color = colors[0]  # Green
            elif aspect_ratio < 0.5:  # Tall = likely table leg
                component_type = "table_leg"
                color = colors[1]  # Blue
            else:
                component_type = "table_component"
                color = colors[min(2 + i, len(colors) - 1)]

            table_components.append({
                'type': component_type,
                'area': area,
                'bbox': [x, y, w, h],
                'aspect_ratio': aspect_ratio
            })

            # Draw contour with colored outline and semi-transparent fill
            cv2.drawContours(vis_image, [contour], -1, color, 3)

            # Create mask for this contour and blend with base image
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 255)

            # Create colored overlay
            overlay = np.zeros_like(base_image)
            overlay[contour_mask > 0] = color

            # Blend with original image (30% opacity)
            cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0, vis_image)

            # Add label
            cv2.putText(vis_image, f"{component_type} ({area:.0f}px²)",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Find and highlight table surface (use center area)
        mask_center_x = mask.shape[1] // 2
        mask_width = mask.shape[1] // 3  # Use 1/3 of image width as overlay width estimate

        surface_info = find_table_surface_from_mask(mask, mask_width, mask_center_x)
        if surface_info['surface_y'] is not None:
            surface_y = surface_info['surface_y']
            # Draw horizontal line showing detected table surface
            cv2.line(vis_image, (0, surface_y), (vis_image.shape[1], surface_y),
                    (0, 0, 255), 3)  # Red line
            cv2.putText(vis_image, f"Table Surface (Y={surface_y})",
                       (20, surface_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Table Detection Analysis:", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, "Green: Table Surface", (20, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, "Blue: Table Legs", (20, legend_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(vis_image, "Red Line: Detected Surface", (20, legend_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save visualization
        cv2.imwrite(output_path, vis_image)
        logger.info(f"Table detection visualization saved to: {output_path}")

        analysis_result = {
            'total_components': len(table_components),
            'components': table_components,
            'surface_y': surface_info['surface_y'],
            'surface_confidence': surface_info['confidence']
        }

        logger.info(f"Detected {len(table_components)} table components")
        for comp in table_components:
            logger.info(f"  {comp['type']}: {comp['area']:.0f}px², aspect ratio: {comp['aspect_ratio']:.2f}")

        return analysis_result

    except Exception as e:
        logger.error(f"Error creating table visualization: {e}")
        return None


def visualize_calculation_areas(base_image_path, mask_path, output_path):
    """
    Creates a visualization showing the exact areas used for ratio calculations.

    Args:
        base_image_path: Path to the original base image
        mask_path: Path to the SAM mask
        output_path: Path to save the visualization

    Returns:
        dict: Information about the detected areas
    """
    logger.info("--- Creating calculation areas visualization ---")
    try:
        # Load base image and mask
        base_image = cv2.imread(base_image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if base_image is None or mask is None:
            logger.error("Could not load base image or mask")
            return None

        # Create visualization image
        vis_image = base_image.copy()

        # Find SAM mask overall bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(largest_contour)

            # Draw overall mask bounding box in BLUE
            cv2.rectangle(vis_image, (x_mask, y_mask), (x_mask + w_mask, y_mask + h_mask),
                         (255, 0, 0), 3)  # Blue
            cv2.putText(vis_image, f"SAM Mask Area: {w_mask}x{h_mask}",
                       (x_mask, y_mask - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Find table surface detection area (center area)
            table_center_x = x_mask + (w_mask // 2)
            estimated_overlay_width = min(w_mask // 2, 300)
            roi_half_width = estimated_overlay_width // 2
            roi_left = max(0, table_center_x - roi_half_width)
            roi_right = min(mask.shape[1], table_center_x + roi_half_width)

            # Draw center detection area in GREEN
            cv2.rectangle(vis_image, (roi_left, y_mask), (roi_right, y_mask + h_mask),
                         (0, 255, 0), 2)  # Green
            cv2.putText(vis_image, f"Center Detection Area: {roi_right-roi_left}px wide",
                       (roi_left, y_mask + h_mask + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Find and draw table surface
            surface_info = find_table_surface_from_mask(mask, estimated_overlay_width, table_center_x)
            if surface_info['surface_y'] is not None:
                surface_y = surface_info['surface_y']

                # Draw table surface line in RED
                cv2.line(vis_image, (roi_left, surface_y), (roi_right, surface_y), (0, 0, 255), 3)
                cv2.putText(vis_image, f"Table Surface Y={surface_y}",
                           (roi_left, surface_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Calculate and draw intended overlay placement area
                placement_height = h_mask // 2
                placement_y = surface_y - placement_height

                # Calculate overlay dimensions using JSON ratios
                # Simulate the ratio calculation
                json_ratio_w = 600 / 1150  # JSON overlay/base width ratio
                json_ratio_h = 1000 / 2280  # JSON overlay/base height ratio

                intended_overlay_w = w_mask * json_ratio_w
                intended_overlay_h = h_mask * json_ratio_h

                # Draw intended overlay area in YELLOW
                overlay_left = int(table_center_x - intended_overlay_w // 2)
                overlay_right = int(table_center_x + intended_overlay_w // 2)
                overlay_top = int(surface_y - intended_overlay_h)
                overlay_bottom = surface_y

                cv2.rectangle(vis_image, (overlay_left, overlay_top), (overlay_right, overlay_bottom),
                             (0, 255, 255), 2)  # Yellow
                cv2.putText(vis_image, f"Intended Overlay: {intended_overlay_w:.0f}x{intended_overlay_h:.0f}",
                           (overlay_left, overlay_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Add center line
                cv2.line(vis_image, (table_center_x, y_mask), (table_center_x, y_mask + h_mask),
                        (255, 255, 255), 1)  # White center line
                cv2.putText(vis_image, f"Center X={table_center_x}",
                           (table_center_x + 5, y_mask + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add legend
            legend_y = 30
            cv2.putText(vis_image, "Calculation Areas Visualization:", (20, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_image, "Blue: Overall SAM Mask Area (Base for ratios)", (20, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(vis_image, "Green: Center Detection Area", (20, legend_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(vis_image, "Red: Detected Table Surface", (20, legend_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis_image, "Yellow: Calculated Overlay Placement", (20, legend_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Add ratio information
            cv2.putText(vis_image, f"JSON Ratios: W={json_ratio_w:.3f}, H={json_ratio_h:.3f}",
                       (20, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save visualization
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Calculation areas visualization saved to: {output_path}")

            result = {
                'mask_area': [x_mask, y_mask, w_mask, h_mask],
                'center_detection_area': [roi_left, y_mask, roi_right - roi_left, h_mask],
                'table_surface_y': surface_info['surface_y'] if surface_info['surface_y'] is not None else None,
                'intended_overlay_size': [intended_overlay_w, intended_overlay_h],
                'json_ratios': [json_ratio_w, json_ratio_h]
            }

            logger.info(f"Mask area: {w_mask}x{h_mask} at ({x_mask}, {y_mask})")
            logger.info(f"Center detection area: {roi_right-roi_left}px wide")
            logger.info(f"Intended overlay size: {intended_overlay_w:.0f}x{intended_overlay_h:.0f}")

            return result

        else:
            logger.error("No contours found in mask")
            return None

    except Exception as e:
        logger.error(f"Error creating calculation areas visualization: {e}")
        return None