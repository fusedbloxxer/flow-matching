from dataclasses import dataclass
from typing import Tuple

import mlflow
import torch

from einops import reduce, repeat
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from torchvision.utils import make_grid
from tqdm import trange

from .config import LogParamConfig
from .flow import FlowPath, Sampleable
from .model import AutoEncoder, FlowModel
from .ode import ODE
from .utils import plot_batch


@dataclass(kw_only=True)
class Trainer:
    logger: LogParamConfig
    device: torch.device
    batch_size: int
    drop: float
    steps: int
    lr: float

    p_data: Sampleable[Tuple[Tensor, Tensor]]
    p_init: Sampleable[Tensor]
    f_model: FlowModel
    f_path: FlowPath
    vae: AutoEncoder

    def __post_init__(self) -> None:
        self.optim = AdamW(self.f_model.parameters(), lr=self.lr)
        self.ode = ODE(model=self.f_model)

    def train(self) -> None:
        pbar = trange(self.steps, desc="steps")

        self.f_model.to(self.device)
        self.vae.to(self.device)

        self.vae.requires_grad_(False)
        self.f_model.train()
        self.vae.eval()

        for step in pbar:
            x, y = self.p_data.sample(self.batch_size)
            x = x.to(self.device)
            y = y.to(self.device)

            drop_mask = torch.rand_like(y.float(), device=self.device) <= self.drop
            y[drop_mask] = self.f_model.class_embedding.padding_idx

            t = torch.rand((self.batch_size, 1), device=self.device)
            x_0 = self.p_init.sample(self.batch_size).flatten(1).to(self.device)
            x_1_latent = self.vae.encode(x).flatten(1)
            x_t = self.f_path.flow(x_0, x_1_latent, t)

            v_true = self.f_path.cond_vector_field(x_1=x_1_latent, x_t=x_t, t=t)
            v_pred = self.f_model.forward(x=x_t, t=t, y=y)

            loss = mse_loss(v_pred, v_true)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            if step % self.logger.step_interval == 0:
                pbar.set_description(f"{loss.item() = :.2f}")

                # Sample data points for all classes
                bs = self.logger.batch_size
                t = torch.linspace(0, 1, self.logger.sample_steps, device=self.device)
                y = torch.arange(0, 10, device=self.device)
                y = repeat(y, "y -> (y b)", b=bs)

                # Sample initial noise
                x_0 = self.p_init.sample(y.size(0)).flatten(1).to(self.device)

                # Simulate ODE
                for w in [0.0, 1.5, 2.5, 3.5, 5.0]:
                    x_1_latents = self.ode.sample(x_0, t, y, w)
                    x_1_images = self.latent_to_grid(x_1_latents)
                    f = plot_batch(x_1_images, bs, show=False)
                    mlflow.log_figure(f, "ode_sample_{:05d}_{}.png".format(step, w))
            mlflow.log_metric("train_loss", value=loss.item(), step=step)

        self.f_model.eval()
        pbar.close()

    def latent_to_grid(self, x_1_latent: Tensor) -> Tensor:
        x_1 = x_1_latent.reshape((-1, 4, 4, 4))
        x_1 = self.vae.decode(x=x_1)
        x_1 = reduce(x_1, "b c h w -> b 1 h w", "mean")
        x_1 = make_grid(x_1.cpu(), nrow=self.logger.batch_size)
        return x_1
