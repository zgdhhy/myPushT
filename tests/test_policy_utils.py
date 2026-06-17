import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mypusht.policies.utils import image_to_chw_float, lowdim_from_sample


def test_lowdim_from_sample_accepts_short_and_observation_keys():
    sample = {
        "observation.state": np.ones(5, dtype=np.float32),
        "observation.mocap_xy": np.ones(2, dtype=np.float32) * 2,
        "observation.object_pose": np.ones(3, dtype=np.float32) * 3,
        "observation.goal_pose": np.ones(3, dtype=np.float32) * 4,
    }

    lowdim = lowdim_from_sample(sample)

    assert tuple(lowdim.shape) == (13,)
    assert torch.is_tensor(lowdim)


def test_image_to_chw_float_normalizes_hwc_uint8():
    image = np.full((4, 5, 3), 255, dtype=np.uint8)

    chw = image_to_chw_float(image)

    assert tuple(chw.shape) == (3, 4, 5)
    assert torch.allclose(chw, torch.ones_like(chw))
