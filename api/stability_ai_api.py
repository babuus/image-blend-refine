from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc
import grpc
from stability_sdk import client
import io # Added import for io

import config
from logger_config import logger

def run_stability_ai_blending_approach():
    """
    Applies Stability AI's image-to-image blending to the geometrically blended image.
    """
    logger.info("--- Running Stability AI Blending Approach ---")
    try:
        # Load the geometrically blended image
        img_for_stability = Image.open(config.LOGIC_OUTPUT_PATH).convert("RGB")

        # Set up Stability AI client
        stability_api = client.StabilityInference(
            key=config.STABILITY_API_KEY,
            host=config.STABILITY_HOST,
            verbose=True # Set to False for less verbose output
        )

        # Create the image-to-image request
        answers = stability_api.generate(
            prompt="seamlessly integrate container into rack, realistic, high quality",
            init_image=img_for_stability, # Pass the PIL Image object directly
            start_schedule=config.STABILITY_DENOISING_STRENGTH, # Denoising strength
            # Other parameters like steps, cfg_scale, engine can be added if needed
            # engine="stable-diffusion-v1-5" is the default for StabilityInference if not specified
        )

        # Process the response
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    logger.warning("Stability AI request activated safety filters and could not be processed.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img_blended = Image.open(io.BytesIO(artifact.binary))
                    img_blended.save(config.STABILITY_OUTPUT_PATH)
                    logger.info(f"Successfully created blended image with Stability AI: '{config.STABILITY_OUTPUT_PATH}'")
                    return # Exit after first image is processed
        
        logger.error("No image artifact found in Stability AI response.")

    except Exception as e:
        logger.error(f"Error during Stability AI blending: {e}")