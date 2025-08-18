import logging
import os

from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    return logging.getLogger("app")