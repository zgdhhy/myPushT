from dataclasses import dataclass
import json
from pathlib import Path

from mypusht.paths import CONFIGS_DIR, RAW_EPISODES_DIR, SO100_XML_PATH

XML_PATH = SO100_XML_PATH
RAW_DIR = RAW_EPISODES_DIR
MAPPING_PATH = CONFIGS_DIR / "controller_mapping.json"

OBS_SHAPE = (224, 224, 3)
ACTUATOR_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")


@dataclass(frozen=True)
class ControllerMapping:
    axis_left_x: int
    axis_left_y: int
    button_record: int
    button_reset: int
    button_exit: int
    button_discard: int
    deadzone: float
    move_speed: float
    fps: int
    workspace_x: tuple[float, float] = (0.05, 0.45)
    workspace_y: tuple[float, float] = (-0.25, 0.25)


def _range_from_mapping(raw: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = raw.get(key, default)
    if len(value) != 2:
        raise ValueError(f"{key} must contain exactly two values")
    low, high = float(value[0]), float(value[1])
    if low >= high:
        raise ValueError(f"{key} lower bound must be smaller than upper bound")
    return low, high



def load_mapping(path: Path = MAPPING_PATH) -> ControllerMapping:
    if path is None:
        path = MAPPING_PATH

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return ControllerMapping(
        axis_left_x=int(raw["axis_left_x"]),
        axis_left_y=int(raw["axis_left_y"]),
        button_record=int(raw["button_record"]),
        button_reset=int(raw["button_reset"]),
        button_exit=int(raw["button_exit"]),
        button_discard=int(raw["button_discard"]),
        deadzone=float(raw["deadzone"]),
        move_speed=float(raw["move_speed"]),
        fps=int(raw["fps"]),
        workspace_x=_range_from_mapping(raw, "workspace_x", (0.05, 0.45)),
        workspace_y=_range_from_mapping(raw, "workspace_y", (-0.25, 0.25)),
    )
