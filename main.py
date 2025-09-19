import os
import sys # Import sys for exiting

import config # Import config as a module to access and modify its attributes
from core.fitter import run_logic_approach, run_logic_only_approach

from logger_config import logger

def select_example():
    """
    Prompts the user to select an example directory.
    """
    while True:
        logger.info("\nAvailable examples:")
        for i, example_name in enumerate(config.AVAILABLE_EXAMPLES.keys()):
            logger.info(f"{i + 1}: {example_name}")

        choice = input("Enter the number of the example to use: ")
        try:
            choice_idx = int(choice) - 1
            example_names = list(config.AVAILABLE_EXAMPLES.keys())
            if 0 <= choice_idx < len(example_names):
                selected_example_name = example_names[choice_idx]
                return selected_example_name, config.AVAILABLE_EXAMPLES[selected_example_name]
            else:
                logger.warning("Invalid choice. Please enter a valid number.")
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")

def main():
    """Main function to drive the script."""
    os.chdir(config.CURRENT_DIR)

    # Select example
    selected_example_name, selected_example_dir = select_example()
    if not selected_example_dir:
        logger.error("No example selected. Exiting.")
        sys.exit(1) # Exit if no example is selected

    # Dynamically set paths in config based on selected example
    config.BASE_IMAGE_PATH = os.path.join(selected_example_dir, "base_image.jpg")
    config.OVERLAY_IMAGE_PATH = os.path.join(selected_example_dir, "overlay_image.png")
    config.MEASUREMENTS_PATH = os.path.join(selected_example_dir, "measurements.json")
    config.LOGIC_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, f"{selected_example_name}_logic.png")
    config.STABILITY_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, f"{selected_example_name}_stability_blended.png")
    config.DEBUG_IMAGE_PATH = os.path.join(config.OUTPUT_DIR, f"{selected_example_name}_debug_quadrilateral.png")
    config.MASK_PATH = os.path.join(config.OUTPUT_DIR, f"{selected_example_name}_mask.png")
    config.COMPARISON_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, f"{selected_example_name}_comparison.png")

    while True:
        logger.info("\nChoose an approach:")
        logger.info("1: Logic (Geometric Blending) with Stability AI Refinement")
        logger.info("2: Logic only (Geometric Blending)")
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            run_logic_approach()
            break
        elif choice == '2':
            run_logic_only_approach()
            break
        else:
            logger.warning("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()