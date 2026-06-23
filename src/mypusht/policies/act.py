import torch
from torch import nn
import torch.nn.functional as F

from collections import deque

from .utils import image_to_chw_float, lowdim_from_sample

class ImageEncoder(nn.Module):

    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ACT(nn.Module):
    """Action Chunking Transformer with a CVAE action encoder.

    The training path follows ACT's core structure:
        posterior z = Encoder([CLS], qpos, action_chunk)
        action_chunk = Decoder(action_queries, visual/state condition, z)

    At inference time, the policy uses z = 0, which is the standard ACT
    deterministic rollout path.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        num_cams: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        image_feature_dim: int = 64,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if obs_horizon <= 0:
            raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.num_cams = int(num_cams)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.image_feature_dim = int(image_feature_dim)
        self.latent_dim = int(latent_dim)
        self.kl_weight = float(kl_weight)
        self.dropout = float(dropout)

        self.image_encoder = ImageEncoder(out_dim=image_feature_dim, dropout=dropout)
        self.image_proj = nn.Linear(image_feature_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        self.obs_time_pos = nn.Parameter(torch.zeros(1, obs_horizon, 1, d_model))
        self.cam_pos = nn.Parameter(torch.zeros(1, 1, num_cams, d_model))
        self.state_pos = nn.Parameter(torch.zeros(1, obs_horizon, d_model))
        self.latent_condition_pos = nn.Parameter(torch.zeros(1, 1, d_model))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.posterior_state_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.action_pos = nn.Parameter(torch.zeros(1, action_horizon, d_model))

        self.action_queries = nn.Parameter(torch.zeros(1, action_horizon, d_model))
        self.latent_proj = nn.Linear(latent_dim, d_model)

        posterior_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(
            posterior_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False
        )
        self.latent_mu = nn.Linear(d_model, latent_dim)
        self.latent_logvar = nn.Linear(d_model, latent_dim)

        condition_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.condition_encoder = nn.TransformerEncoder(
            condition_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for param in (
            self.cls_token,
            self.action_queries,
            self.obs_time_pos,
            self.cam_pos,
            self.state_pos,
            self.latent_condition_pos,
            self.posterior_state_pos,
            self.action_pos,
        ):
            nn.init.normal_(param, std=0.02)


    def _encode_posterior(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = states.shape[0]
        current_state = states[:, -1]
        _cls = self.cls_token.expand(b, -1, -1)
        state_token = self.state_proj(current_state).unsqueeze(1) + self.posterior_state_pos
        action_tokens = self.action_proj(actions) + self.action_pos

        tokens = torch.cat([_cls, state_token, action_tokens], dim=1)
        encoded = self.posterior_encoder(tokens)
        stats_token = encoded[:, 0]
        mu = self.latent_mu(stats_token)
        logvar = self.latent_logvar(stats_token).clamp(min=-10.0, max=10.0)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def _encode_condition(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        b, t, ncam, c, h, w = images.shape

        flat_images = images.reshape(b * t * ncam, c, h, w)
        image_features = self.image_encoder(flat_images)
        image_tokens = self.image_proj(image_features)
        image_tokens = image_tokens.reshape(b, t, ncam, self.d_model)
        image_tokens = image_tokens + self.obs_time_pos + self.cam_pos
        image_tokens = image_tokens.flatten(start_dim=1, end_dim=2)

        state_tokens = self.state_proj(states) + self.state_pos
        latent_token = self.latent_proj(z).unsqueeze(1) + self.latent_condition_pos

        tokens = torch.cat([latent_token, state_tokens, image_tokens], dim=1)
        return self.condition_encoder(tokens)

    def _decode(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        b = condition_tokens.shape[0]
        queries = self.action_queries.expand(b, -1, -1) + self.action_pos
        decoded = self.decoder(tgt=queries, memory=condition_tokens)
        return self.action_head(decoded)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        actions: torch.Tensor | None = None,
    ):
        """Predict an action chunk.

        Args:
            states: [B, obs_horizon, state_dim]
            images: [B, obs_horizon, num_cams, 3, H, W]
            actions: [B, action_horizon, action_dim], only used during training

        Returns:
            If actions is provided: (action_pred, kl_loss)
            Otherwise: action_pred
        """

        if actions is not None:
            z, mu, logvar = self._encode_posterior(states, actions)
            condition_tokens = self._encode_condition(states, images, z)
            action_pred = self._decode(condition_tokens)
            kl = self._kl_to_standard_normal(mu, logvar)
            return action_pred, kl

        z = torch.zeros(
            states.shape[0],
            self.latent_dim,
            device=states.device,
            dtype=states.dtype,
        )
        condition_tokens = self._encode_condition(states, images, z)
        return self._decode(condition_tokens)

    def compute_loss(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_pred, kl = self.forward(states, images, actions)
        mse = F.mse_loss(action_pred, actions)
        loss = mse + self.kl_weight * kl
        return loss, mse, kl

    def checkpoint_kwargs(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "action_horizon": self.action_horizon,
            "num_cams": self.num_cams,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "image_feature_dim": self.image_feature_dim,
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
            "dropout": self.dropout,
        }



class ACTPolicy:
    def __init__(self, ckpt_path, device, exec_horizon=4):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.device = device
        self.exec_horizon = exec_horizon

        self.model = ACT(**ckpt["kwargs"]).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.state_mean = ckpt["state_mean"].to(device)
        self.state_std = ckpt["state_std"].to(device)
        self.action_mean = ckpt["action_mean"].to(device)
        self.action_std = ckpt["action_std"].to(device)

        self.obs_horizon = self.model.obs_horizon

        self.obs_history = deque(maxlen=self.obs_horizon)
        self.action_history = deque()

    def reset(self):
        self.obs_history.clear()
        self.action_history.clear()
    
    def _make_obs_batch(self):
        history = list(self.obs_history)
        if len(history) < self.obs_horizon:
            padding = [history[0]] * (self.obs_horizon - len(history))
            history = padding + history
        states = torch.stack(
            [lowdim_from_sample(sample) for sample in history], 
            dim=0
        ).unsqueeze(0).to(self.device)
        states = (states - self.state_mean) / self.state_std

        images = []
        for sample in history:
            images.append(
                torch.stack(
                    [image_to_chw_float(sample["images"][cam]) for cam in ["cam_top", "cam_side"]],
                    dim=0,
                )
            )
        images = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
        return states, images

    @torch.no_grad()
    def predict(self, obs):
        self.obs_history.append(obs)
        if len(self.action_history) == 0:
            states, images = self._make_obs_batch()
            pred_action = self.model(states, images)

            pred_action = pred_action * self.action_std + self.action_mean
            pred_action = pred_action.squeeze(0).detach().cpu()
            
            for action in pred_action[: self.exec_horizon]:
                self.action_history.append(action)

        return self.action_history.popleft()
