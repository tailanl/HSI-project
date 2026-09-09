"""Cosine diffusion and explicit motion normalization, without pretrained assets."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from .contracts import MOTION_DIM


class MotionDiffusion(nn.Module):
    def __init__(self, steps: int = 1000, mean: Tensor | None = None, scale: Tensor | None = None):
        super().__init__()
        if steps < 2:
            raise ValueError("diffusion needs at least two steps")
        if (mean is None) != (scale is None):
            raise ValueError("normalization requires both mean and scale")
        self.identity_normalization = mean is None
        mean = torch.zeros(MOTION_DIM) if mean is None else torch.as_tensor(mean).float()
        scale = torch.ones(MOTION_DIM) if scale is None else torch.as_tensor(scale).float()
        if mean.shape != (MOTION_DIM,) or scale.shape != (MOTION_DIM,):
            raise ValueError("normalization statistics must have 135 channels")
        if not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("normalization must be finite with positive scale")
        self.register_buffer("mean",mean)
        self.register_buffer("scale",scale)
        time = torch.linspace(0,steps,steps+1,dtype=torch.float64)/steps
        alpha_bar = torch.cos((time+0.008)/1.008*math.pi/2).square()
        alpha_bar = alpha_bar/alpha_bar[0]
        betas = (1-alpha_bar[1:]/alpha_bar[:-1]).clamp(1e-8,0.999)
        self.register_buffer("alpha_bar",(1-betas).cumprod(0).float())

    @property
    def steps(self):
        return len(self.alpha_bar)

    def normalize(self, motion):
        return (motion-self.mean.to(motion))/self.scale.to(motion)

    def denormalize(self, motion):
        return motion*self.scale.to(motion)+self.mean.to(motion)

    def add_noise(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        if clean.shape != noise.shape or timesteps.shape != clean.shape[:1]:
            raise ValueError("diffusion noise/timestep shape mismatch")
        if timesteps.dtype != torch.long or bool(((timesteps < 0) | (timesteps >= self.steps)).any()):
            raise ValueError("timestep outside schedule")
        alpha = self.alpha_bar[timesteps].to(clean)[:,None,None]
        return alpha.sqrt()*clean+(1-alpha).sqrt()*noise

    def inference_steps(self, count: int) -> list[int]:
        if not isinstance(count,int) or not 2 <= count <= self.steps:
            raise ValueError("sampling step count must be in [2,diffusion_steps]")
        return torch.linspace(self.steps-1,0,count).round().long().tolist()

    def ddim_step(self, noisy: Tensor, predicted_clean: Tensor, current: int, following: int) -> Tensor:
        """Deterministic DDIM (eta=0); follow=-1 returns the clean endpoint."""
        if following == -1:
            return predicted_clean
        if not 0 <= following < current < self.steps:
            raise ValueError("DDIM indices must descend")
        alpha = self.alpha_bar[current].to(noisy)
        next_alpha = self.alpha_bar[following].to(noisy)
        noise = (noisy-alpha.sqrt()*predicted_clean)/(1-alpha).sqrt().clamp_min(1e-8)
        return next_alpha.sqrt()*predicted_clean+(1-next_alpha).sqrt()*noise
