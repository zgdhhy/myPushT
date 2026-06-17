from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SIMPLE_XML_PATH = ASSETS_DIR / "envs" / "simple_PushT" / "simple_PushT.xml"
SO100_XML_PATH = ASSETS_DIR / "envs" / "so100_PushT" / "so100_PushT.xml"
MODEL_DIR = ASSETS_DIR / "model"
RAW_EPISODES_DIR = OUTPUTS_DIR / "data" / "raw_episodes"
LEROBOT_DATASET_DIR = OUTPUTS_DIR / "data" / "lerobot_dataset"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "eval"
