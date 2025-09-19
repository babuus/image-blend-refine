import cv2
import numpy as np
import json
import requests
import base64

import config
from utils.image_processing import get_file_content_as_base64
from logger_config import logger

def run_modelslab_inpaint_approach():
    """(Implementation is correct and remains the same)"""
    logger.info("--- Running ModelLabs Inpaint (Diffusion API) Approach ---")
    try:
        logger.info("1. Loading images and measurements...")
        rack_img = cv2.imread(config.BASE_IMAGE_PATH)
        if rack_img is None:
            logger.error(f"Could not load base image at {config.BASE_IMAGE_PATH}")
            return
        with open(config.MEASUREMENTS_PATH, 'r') as f:
            measurements = json.load(f)

        logger.info("2. Creating mask...")
        tl = measurements['base']['position']['top_left']
        br = measurements['base']['position']['bottom_right']
        mask = np.zeros(rack_img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, tuple(tl), tuple(br), 255, -1)

        logger.info("3. Encoding images to base64...")
        init_image_b64 = get_file_content_as_base64(config.BASE_IMAGE_PATH)
        _, mask_bytes = cv2.imencode('.png', mask)
        mask_image_b64 = base64.b64encode(mask_bytes).decode("utf-8")
        logger.info("   - Base64 encoding complete.")

        payload = {
            "key": config.MODELSLAB_API_KEY,
            "prompt": "A shipping container perfectly placed inside the empty rack slot, matching the scene's perspective and lighting.",
            "negative_prompt": "blurry, distorted, unrealistic",
            "init_image": init_image_b64,
            "mask_image": mask_image_b64,
            "width": rack_img.shape[1],
            "height": rack_img.shape[0],
            "samples": 1,
            "num_inference_steps": 30,
            "safety_checker": "no",
            "guidance_scale": 5.0, # Corrected value
            "strength": 0.8,
            "base64": "true" 
        }

        logger.info(f"4. Calling diffusion API at {config.MODELSLAB_API_URL}")
        response = requests.post(config.MODELSLAB_API_URL, json=payload, timeout=60) # Added timeout
        logger.info(f"5. Received response from API. HTTP Status Code: {response.status_code}")

        data = response.json()

        if response.ok and data.get("status") == 'success' and data.get("output"):
            logger.info("6. Fetching and decoding the result...")
            output_url = data["output"][0]
            logger.info(f"   - Fetching base64 string from: {output_url}")
            base64_response = requests.get(output_url, timeout=120) # Fetch the base64 string
            base64_response.raise_for_status() # Raise an exception for bad status codes
            output_b64 = base64_response.text # Get the base64 string from the response text

            img_bytes = base64.b64decode(output_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            output_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            cv2.imwrite(config.MODELSLAB_INPAINT_OUTPUT_PATH, output_img)
            logger.info(f"Successfully created '{config.MODELSLAB_INPAINT_OUTPUT_PATH}'")
        else:
            logger.warning("API call did not return a successful result.")
            logger.warning(f"Full API Response: {json.dumps(data, indent=2)}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling API: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
