from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os


# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

#MODEL_NAME = "microsoft/swin-large-patch4-window7-224-in22k"
MODEL_NAME = "google/vit-base-patch16-224-in21k"

REPO_ID = os.getenv("REPO_ID") 
MODEL = os.getenv("MODEL")
REVERSE = os.getenv("REVERSE")

