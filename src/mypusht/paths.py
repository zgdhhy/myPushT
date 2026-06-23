from __future__ import annotations

from importlib import resources
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_ASSETS_DIR = Path(resources.files("mypusht") / "assets")

if (PACKAGE_ROOT.parents[1] / "pyproject.toml").exists():
    PROJECT_ROOT = PACKAGE_ROOT.parents[1]
else:
    PROJECT_ROOT = Path.cwd()

ASSETS_DIR = PACKAGE_ASSETS_DIR
EXTERNAL_ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 存放xml文件的目录
SIMPLE_XML_PATH = ASSETS_DIR / "envs" / "simplePushT" / "simple_pusht.xml"
SO100_XML_PATH = ASSETS_DIR / "envs" / "so100PushT" / "so100_PushT.xml"
# 存放训练好的模型的目录
MODEL_DIR = EXTERNAL_ASSETS_DIR / "model"
# 存放原始episode数据的目录
RAW_EPISODES_DIR = OUTPUTS_DIR / "data" / "raw_episodes"
# 存放转换后的lerobot数据集的目录
LEROBOT_DATASET_DIR = OUTPUTS_DIR / "data" / "lerobot_dataset"
# 存放评估结果的目录
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "eval"
