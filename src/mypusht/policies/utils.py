import torch

LOWDIM_KEYS = [
    "state",
    "mocap_xy",
    "object_pose",
    "goal_pose",
]

def image_to_chw_float(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {tuple(x.shape)}")

    # LeRobot / video backends may return CHW or HWC depending on version.
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        chw = x.detach()
    else:
        chw = x.detach().permute(2, 0, 1)

    chw = chw.to(dtype=torch.float32).contiguous()
    if chw.max().item() > 2.0:
        chw = chw / 255.0
    return chw


def resolve_sample_key(sample, key):
    if key in sample:
        return key
    observation_key = f"observation.{key}"
    if observation_key in sample:
        return observation_key
    if key.startswith("observation."):
        short_key = key.removeprefix("observation.")
        if short_key in sample:
            return short_key
    raise KeyError(f"missing key: {key}")


def lowdim_from_sample(sample):
    pieces = [
        torch.as_tensor(sample[resolve_sample_key(sample, key)], dtype=torch.float32).reshape(-1)
        for key in LOWDIM_KEYS
    ]
    return torch.cat(pieces, dim=0)
