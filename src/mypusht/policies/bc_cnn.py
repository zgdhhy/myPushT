import torch
import torch.nn as nn
import numpy as np

from mypusht.policies.utils import image_to_chw_float, lowdim_from_sample


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.proj(self.conv(image))


class BCCNN(nn.Module):
    def __init__(self, state_dim, action_dim=2, image_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.top_encoder = ImageEncoder(out_dim=image_dim, dropout=dropout)
        self.side_encoder = ImageEncoder(out_dim=image_dim, dropout=dropout)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(image_dim * 2 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, cam_top, cam_side, state):
        top_feat = self.top_encoder(cam_top)
        side_feat = self.side_encoder(cam_side)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([top_feat, side_feat, state_feat], dim=-1))
    

class BCCNNPolicy:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.device = device
        self.model = BCCNN(
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
        ).to(device)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        
        self.state_mean = ckpt["state_mean"].to(device)
        self.state_std = ckpt["state_std"].to(device)
        self.action_mean = ckpt["action_mean"].to(device)
        self.action_std = ckpt["action_std"].to(device)

    @torch.no_grad()
    def predict(self, obs):
        cam_top = image_to_chw_float(obs["images"]["cam_top"]).to(self.device).unsqueeze(0)
        cam_side = image_to_chw_float(obs["images"]["cam_side"]).to(self.device).unsqueeze(0)
        state = lowdim_from_sample(obs).to(self.device).unsqueeze(0)
        
        state_norm = (state - self.state_mean) / self.state_std
        action_norm = self.model(cam_top, cam_side, state_norm)
        action = action_norm * self.action_std + self.action_mean
        return action.squeeze(0).cpu().numpy().astype(np.float32)
