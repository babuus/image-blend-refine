import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline
from controlnet_aux import MidasDetector, CannyDetector
import json

import config
from logger_config import logger
from utils.image_processing import get_object_bounding_box

class ControlNetProcessor:
    def __init__(self):
        self.depth_detector = None
        self.canny_detector = None
        self.depth_controlnet_pipeline = None
        self.inpaint_controlnet_pipeline = None
        self.device = "cpu"  # Force CPU usage as specified by user
        logger.info(f"ControlNet processor initialized on device: {self.device}")

    def load_depth_pipeline(self):
        """Load ControlNet pipeline for depth conditioning"""
        if self.depth_controlnet_pipeline is None:
            logger.info("Loading ControlNet depth pipeline for CPU...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float32  # Use float32 for CPU
            )

            self.depth_controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            # Enable memory efficient attention for CPU
            self.depth_controlnet_pipeline.enable_attention_slicing()

            self.depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            logger.info("ControlNet depth pipeline loaded successfully")

    def load_inpaint_pipeline(self):
        """Load ControlNet pipeline for inpainting"""
        if self.inpaint_controlnet_pipeline is None:
            logger.info("Loading ControlNet inpaint pipeline for CPU...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint",
                torch_dtype=torch.float32  # Use float32 for CPU
            )

            self.inpaint_controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            # Enable memory efficient attention for CPU
            self.inpaint_controlnet_pipeline.enable_attention_slicing()

            logger.info("ControlNet inpaint pipeline loaded successfully")

    def generate_depth_map(self, image_path):
        """Generate depth map from input image"""
        image = Image.open(image_path).convert("RGB")
        depth_image = self.depth_detector(image)
        return depth_image

    def create_placement_mask(self, base_image_path, placement_coords):
        """Create mask for the placement area"""
        base_image = cv2.imread(base_image_path)
        mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)

        # Create mask for the placement area
        x, y, w, h = placement_coords
        mask[y:y+h, x:x+w] = 255

        # Apply Gaussian blur for smoother blending
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return Image.fromarray(mask)

    def run_depth_controlled_placement(self, product_size="default"):
        """
        Generate realistic product placement using depth conditioning
        """
        logger.info("Running ControlNet depth-controlled placement...")

        try:
            self.load_depth_pipeline()

            # Load base image and generate depth map
            base_image = Image.open(config.BASE_IMAGE_PATH).convert("RGB")
            depth_image = self.generate_depth_map(config.BASE_IMAGE_PATH)

            # Get placement coordinates from measurements
            with open(config.MEASUREMENTS_PATH, 'r') as f:
                measurements = json.load(f)

            # Adjust prompt based on product size
            size_prompts = {
                "small": "small TV mounted on wall, 42 inch screen, realistic lighting",
                "large": "large TV mounted on wall, 55 inch screen, realistic lighting",
                "default": "TV mounted on wall, realistic lighting, natural shadows"
            }

            prompt = size_prompts.get(product_size, size_prompts["default"])
            negative_prompt = "blurry, distorted, unrealistic, low quality, artifacts"

            # Generate image with depth conditioning (CPU optimized)
            result = self.depth_controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=10,  # Reduced steps for CPU
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                height=512,  # Fixed smaller resolution for CPU
                width=512
            )

            output_path = config.LOGIC_OUTPUT_PATH.replace('.png', f'_controlnet_depth_{product_size}.png')
            result.images[0].save(output_path)
            logger.info(f"ControlNet depth placement saved: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error in depth-controlled placement: {e}")
            return None

    def run_inpaint_placement(self, product_size="default"):
        """
        Generate realistic product placement using inpainting
        """
        logger.info("Running ControlNet inpaint placement...")

        try:
            self.load_inpaint_pipeline()

            # Load base image
            base_image = Image.open(config.BASE_IMAGE_PATH).convert("RGB")

            # Get placement coordinates and create mask
            with open(config.MEASUREMENTS_PATH, 'r') as f:
                measurements = json.load(f)

            tl = measurements['base']['position']['top_left']
            br = measurements['base']['position']['bottom_right']
            placement_coords = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]

            mask_image = self.create_placement_mask(config.BASE_IMAGE_PATH, placement_coords)

            # Adjust prompt based on product size
            size_prompts = {
                "small": "modern 42 inch TV on wall mount, sleek design, realistic",
                "large": "large 55 inch TV on wall mount, premium design, realistic",
                "default": "TV on wall mount, modern design, realistic lighting"
            }

            prompt = size_prompts.get(product_size, size_prompts["default"])
            negative_prompt = "blurry, distorted, unrealistic, low quality"

            # Generate inpainted image (CPU optimized)
            result = self.inpaint_controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image.resize((512, 512)),  # Resize for CPU efficiency
                mask_image=mask_image.resize((512, 512)),
                num_inference_steps=10,  # Reduced steps for CPU
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0
            )

            output_path = config.LOGIC_OUTPUT_PATH.replace('.png', f'_controlnet_inpaint_{product_size}.png')
            result.images[0].save(output_path)
            logger.info(f"ControlNet inpaint placement saved: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error in inpaint placement: {e}")
            return None

    def generate_multiple_sizes(self, sizes=["small", "large"]):
        """
        Generate multiple product size variations
        """
        logger.info("Generating multiple product size variations...")
        results = {}

        for size in sizes:
            logger.info(f"Generating {size} size variation...")

            # Generate both depth and inpaint versions
            depth_result = self.run_depth_controlled_placement(size)
            inpaint_result = self.run_inpaint_placement(size)

            results[size] = {
                "depth": depth_result,
                "inpaint": inpaint_result
            }

        logger.info(f"Generated {len(sizes)} size variations")
        return results


def run_controlnet_depth_approach(product_size="default"):
    """
    Main function to run ControlNet depth conditioning approach
    """
    processor = ControlNetProcessor()
    return processor.run_depth_controlled_placement(product_size)


def run_controlnet_inpaint_approach(product_size="default"):
    """
    Main function to run ControlNet inpainting approach
    """
    processor = ControlNetProcessor()
    return processor.run_inpaint_placement(product_size)


def run_controlnet_multi_size_approach():
    """
    Main function to generate multiple product size variations
    """
    processor = ControlNetProcessor()
    return processor.generate_multiple_sizes(["small", "large"])