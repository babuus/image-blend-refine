import os
from logger_config import logger # Import logger for warnings
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# --- Available Examples ---
AVAILABLE_EXAMPLES = {
    "rack_container": os.path.join(CURRENT_DIR, "examples", "rack_container"),
    "tv_wall": os.path.join(CURRENT_DIR, "examples", "tv_wall"), # Images for this example are not yet present
}

# --- File Paths (These will be dynamically set in main.py based on user selection) ---
BASE_IMAGE_PATH = None
OVERLAY_IMAGE_PATH = None
MEASUREMENTS_PATH = None
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')
LOGIC_OUTPUT_PATH = None
STABILITY_OUTPUT_PATH = None
DEBUG_IMAGE_PATH = None
COMPARISON_OUTPUT_PATH = None
SAM_OUTPUT_PATH = None

# Stability AI Configuration
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
if not STABILITY_API_KEY:
    logger.warning("STABILITY_API_KEY environment variable not set. Stability AI functionality may be limited.")
STABILITY_HOST = "grpc.stability.ai:443" # Default gRPC host for Stability AI
STABILITY_DENOISING_STRENGTH = 0.2 # Low denoising for blending
STABILITY_OUTPUT_PATH = os.path.join(CURRENT_DIR, 'output', 'rack_with_container_stability_blended.png')
MASK_PATH = os.path.join(CURRENT_DIR, "output", "mask.png")
DEBUG_IMAGE_PATH = os.path.join(CURRENT_DIR, "output", "debug_quadrilateral.png")