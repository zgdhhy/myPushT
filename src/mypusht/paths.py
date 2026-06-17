from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SIMPLE_XML_PATH = ASSETS_DIR / "xml" / "simple_pusht.xml"
SO100_XML_PATH = ASSETS_DIR / "so100" / "human_env.xml"
MODEL_DIR = ASSETS_DIR / "model"
RAW_EPISODES_DIR = OUTPUTS_DIR / "phase3" / "raw_episodes"
LEROBOT_DATASET_DIR = OUTPUTS_DIR / "phase3" / "lerobot_dataset" / "dataset_v3"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "eval"
