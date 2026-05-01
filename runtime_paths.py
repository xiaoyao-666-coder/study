import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", SCRIPT_DIR))
WORK_DIR = Path(os.environ.get("WORK_DIR", PROJECT_DIR))
DATASET_DIR = Path(os.environ.get("DATASET_DIR", WORK_DIR / "dataset"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", WORK_DIR / "checkpoints"))
CHECKPOINT_NC2023_DIR = Path(os.environ.get("CHECKPOINT_NC2023_DIR", WORK_DIR / "checkpoints_nc2023"))
CHECKPOINT_OTW_DIR = Path(os.environ.get("CHECKPOINT_OTW_DIR", WORK_DIR / "checkpoints_otw"))
PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", WORK_DIR / "plots"))


def ensure_project_on_path():
    project_str = str(PROJECT_DIR)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
    return PROJECT_DIR
