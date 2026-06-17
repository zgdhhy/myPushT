import numpy as np
import pytest

from mypusht.evaluation.action_validation import coerce_action


def test_coerce_action_trims_extra_values():
    action = coerce_action([1.0, 2.0, 3.0], action_dim=2)
    np.testing.assert_allclose(action, [1.0, 2.0])
    assert action.dtype == np.float32


def test_coerce_action_rejects_short_values():
    with pytest.raises(ValueError, match="expected at least 2"):
        coerce_action([1.0], action_dim=2)


def test_coerce_action_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        coerce_action([1.0, float("nan")], action_dim=2)
