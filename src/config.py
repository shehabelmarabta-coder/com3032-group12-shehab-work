from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

RANDOM_STATE = 42
TARGET_COLUMN = "Class"
