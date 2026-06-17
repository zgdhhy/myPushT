import torch
import torch.nn as nn
import numpy as np

from mypusht.policies.utils import lowdim_from_sample


class BCMLP(nn.Module):
    def __init__(self, input_dim, action_dim=2, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class BCMLPPolicy:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.device = device
        self.model = BCMLP(
            input_dim=ckpt["input_dim"],
            action_dim=ckpt["action_dim"],
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.x_mean = ckpt["x_mean"].to(device)
        self.x_std = ckpt["x_std"].to(device)
        self.action_mean = ckpt["action_mean"].to(device)
        self.action_std = ckpt["action_std"].to(device)

    @torch.no_grad()
    def predict(self, obs):
        x = lowdim_from_sample(obs).to(self.device).unsqueeze(0)
        x_norm = (x - self.x_mean) / self.x_std
        action_norm = self.model(x_norm)
        action = action_norm * self.action_std + self.action_mean
        return action.squeeze(0).cpu().numpy().astype(np.float32)
