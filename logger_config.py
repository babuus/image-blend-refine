import logging
import os

LOG_FILE = "app.log"

def setup_logging():
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also log to console
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
