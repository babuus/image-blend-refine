import os

# --- Configuration ---
MODELSLAB_API_KEY = "0rmPNrUet38oJgmV5EFOUXbsEY0vcL9LlcAEBsrf26WSjtFQHeZ70kITPBP2"
MODELSLAB_API_URL = "https://modelslab.com/api/v6/image_editing/inpaint"

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
LOGIC_OUTPUT_PATH = os.path.join(CURRENT_DIR, 'output', 'rack_with_container_logic.png')

# Stability AI Configuration
STABILITY_API_KEY = "sk-rk4W0qSh9nLkqNfqr2Fs7qeuLzoiOFT5ybgdp2PMkbWXCDls"
STABILITY_HOST = "grpc.stability.ai:443" # Default gRPC host for Stability AI
STABILITY_DENOISING_STRENGTH = 0.2 # Low denoising for blending
STABILITY_OUTPUT_PATH = os.path.join(CURRENT_DIR, 'output', 'rack_with_container_stability_blended.png')
MODELSLAB_INPAINT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "output", "rack_with_container_modelslab.png")
MASK_PATH = os.path.join(CURRENT_DIR, "output", "mask.png")
DEBUG_IMAGE_PATH = os.path.join(CURRENT_DIR, "output", "debug_quadrilateral.png")