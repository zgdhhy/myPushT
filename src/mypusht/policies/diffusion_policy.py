import math
import torch
from torch import nn

from collections import deque

from .utils import image_to_chw_float, lowdim_from_sample


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class NoiseScheduler:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device)
        self._build()

    def _build(self):
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_steps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.device = torch.device(device)
        self._build()
        return self

    def _extract(self, values, timesteps, target_shape):
        timesteps = timesteps.to(values.device)
        out = values.index_select(0, timesteps)
        return out.view(-1, *([1] * (len(target_shape) - 1)))

    def q_sample(self, clean_actions, timesteps):
        noise = torch.randn_like(clean_actions)
        alpha_bar = self._extract(self.alpha_bars, timesteps, clean_actions.shape)
        return alpha_bar.sqrt() * clean_actions + (1.0 - alpha_bar).sqrt() * noise, noise

    def step(self, pred_noise, timestep, sample):
        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]

        mean = (
            sample - beta_t / (1.0 - alpha_bar_t).sqrt() * pred_noise
        ) / alpha_t.sqrt()
        if timestep > 0:
            alpha_bar_prev_t = self.alpha_bars[timestep - 1]
            sigma2 = (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * beta_t
            noise = torch.randn_like(sample)
            return mean + sigma2.sqrt() * noise
        return mean

    def ddim_step(self, pred_noise, timestep, next_timestep, sample, eta=0.0):
        alpha_bar_t = self._extract(
            self.alpha_bars,
            torch.tensor([timestep], device=sample.device),
            sample.shape,
        )
        alpha_bar_next = self._extract(
            self.alpha_bars,
            torch.tensor([next_timestep], device=sample.device),
            sample.shape,
        )

        sigma = eta * (
            (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp(min=1e-8)
            * (1.0 - alpha_bar_t / alpha_bar_next.clamp(min=1e-8))
        ).sqrt()

        pred_x0 = (
            sample - (1.0 - alpha_bar_t).sqrt() * pred_noise
        ) / alpha_bar_t.sqrt().clamp(min=1e-8)
        dir_xt = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_next - sigma**2, min=0.0)
        ) * pred_noise

        x_next = alpha_bar_next.sqrt() * pred_x0 + dir_xt
        if eta > 0:
            x_next = x_next + sigma * torch.randn_like(sample)
        return x_next

    def get_ddim_timesteps(self, num_inference_steps):
        step_ratio = self.num_steps // num_inference_steps
        timesteps = (
            torch.arange(0, num_inference_steps, device=self.device) * step_ratio
        )
        timesteps = (self.num_steps - 1 - timesteps).long()
        return timesteps.tolist()


class FiLMResBlock1D(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels * 4),
        )

    def forward(self, x, emb):
        film = self.film(emb)
        scale1, shift1, scale2, shift2 = film.chunk(4, dim=1)
        scale1, shift1 = scale1.unsqueeze(-1), shift1.unsqueeze(-1)
        scale2, shift2 = scale2.unsqueeze(-1), shift2.unsqueeze(-1)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = h * (1 + scale1) + shift1

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = h * (1 + scale2) + shift2

        return x + h


class Conv1DDenoiser(nn.Module):
    def __init__(
        self,
        action_dim,
        cond_dim,
        time_dim=128,
        d_model=256,
        num_blocks=6,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        self.input_conv = nn.Conv1d(
            action_dim, d_model, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [FiLMResBlock1D(d_model, d_model * 4) for _ in range(num_blocks)]
        )

        self.output_norm = nn.GroupNorm(min(32, d_model), d_model)
        self.output_conv = nn.Conv1d(
            d_model, action_dim, kernel_size=3, padding=1
        )
        self.act = nn.SiLU()

    @property
    def num_blocks(self):
        return len(self.blocks)

    def forward(self, x, cond, t):
        x = x.permute(0, 2, 1)

        x = self.input_conv(x)

        t_emb = self.time_mlp(t)
        c_emb = self.cond_proj(cond)
        global_emb = torch.cat([t_emb, c_emb], dim=-1)

        for block in self.blocks:
            x = block(x, global_emb)

        x = self.output_norm(x)
        x = self.act(x)
        x = self.output_conv(x)
        x = x.permute(0, 2, 1)
        return x


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=2,
        obs_horizon=2,
        action_horizon=16,
        num_cams=2,
        d_model=256,
        image_feature_dim=64,
        time_dim=128,
        num_diffusion_steps=100,
        beta_start=1e-4,
        beta_end=0.02,
        dropout=0.1,
        num_denoiser_blocks=6,
    ):
        super().__init__()
        if obs_horizon <= 0:
            raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_cams = num_cams
        self.d_model = d_model
        self.image_feature_dim = image_feature_dim
        self.time_dim = time_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.dropout = dropout
        self.num_denoiser_blocks = num_denoiser_blocks

        self.noise_scheduler = NoiseScheduler(
            num_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.image_encoder = ImageEncoder(
            out_dim=image_feature_dim, dropout=self.dropout
        )
        cond_flat_dim = obs_horizon * (state_dim + num_cams * image_feature_dim)

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_flat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.denoiser = Conv1DDenoiser(
            action_dim=action_dim,
            cond_dim=d_model,
            time_dim=time_dim,
            d_model=d_model,
            num_blocks=num_denoiser_blocks,
        )

    def encode_condition(self, states, images):
        b, t, ncam, c, h, w = images.shape
        flat_images = images.reshape(b * t * ncam, c, h, w)
        image_features = self.image_encoder(flat_images)
        image_features = image_features.reshape(b, t, ncam, -1)

        states_flat = states.flatten(start_dim=1)
        images_flat = image_features.flatten(start_dim=1)
        cond = torch.cat([states_flat, images_flat], dim=-1)
        return self.cond_encoder(cond)

    def forward(self, states, images, noisy_actions, timesteps):
        cond = self.encode_condition(states, images)
        return self.denoiser(noisy_actions, cond, timesteps)

    @torch.no_grad()
    def sample(self, states, images, scheduler=None, num_inference_steps=None):
        scheduler = scheduler or self.noise_scheduler
        scheduler.to(states.device)

        if num_inference_steps is not None and num_inference_steps < scheduler.num_steps:
            return self._sample_ddim(states, images, scheduler, num_inference_steps)

        return self._sample_ddpm(states, images, scheduler)

    @torch.no_grad()
    def _sample_ddpm(self, states, images, scheduler):
        b = states.shape[0]
        actions = torch.randn(
            b, self.action_horizon, self.action_dim, device=states.device
        )

        for timestep in reversed(range(1, scheduler.num_steps)):
            ts = torch.full(
                (b,), timestep, dtype=torch.long, device=states.device
            )
            pred_noise = self.forward(states, images, actions, ts)
            actions = scheduler.step(pred_noise, timestep, actions)
        return actions

    @torch.no_grad()
    def _sample_ddim(self, states, images, scheduler, num_inference_steps):
        b = states.shape[0]
        actions = torch.randn(
            b, self.action_horizon, self.action_dim, device=states.device
        )

        timesteps = scheduler.get_ddim_timesteps(num_inference_steps)

        for i, timestep in enumerate(timesteps):
            ts = torch.full(
                (b,), timestep, dtype=torch.long, device=states.device
            )
            pred_noise = self.forward(states, images, actions, ts)
            next_timestep = (
                timesteps[i + 1] if i + 1 < len(timesteps) else 0
            )
            actions = scheduler.ddim_step(
                pred_noise, timestep, next_timestep, actions, eta=0.0
            )
        return actions

    def checkpoint_kwargs(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "action_horizon": self.action_horizon,
            "num_cams": self.num_cams,
            "d_model": self.d_model,
            "image_feature_dim": self.image_feature_dim,
            "time_dim": self.time_dim,
            "num_diffusion_steps": self.num_diffusion_steps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "dropout": self.dropout,
            "num_denoiser_blocks": self.num_denoiser_blocks,
        }



class DPPolicy:
    def __init__(self, ckpt_path, device, exec_horizon=4):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.device = device
        self.exec_horizon = exec_horizon

        self.model = DiffusionPolicy(**ckpt["kwargs"]).to(device)
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
            pred_action = self.model.sample(states, images)

            pred_action = pred_action * self.action_std + self.action_mean
            pred_action = pred_action.squeeze(0).detach().cpu()
            
            for action in pred_action[: self.exec_horizon]:
                self.action_history.append(action)

        return self.action_history.popleft()