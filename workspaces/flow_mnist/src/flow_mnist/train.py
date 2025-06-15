import torch
import mlflow

from tqdm import trange
from typing import Tuple
from einops import rearrange
from dataclasses import dataclass
from torch import Tensor
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from torchvision.utils import make_grid

from .ode import ODE
from .utils import plot_batch
from .config import LogParamConfig
from .flow import FlowPath, Sampleable
from .model import FlowModel, AutoEncoder


@dataclass(kw_only=True)
class Trainer:
    logger: LogParamConfig
    device: torch.device
    batch_size: int
    steps: int
    lr: float

    p_data: Sampleable[Tuple[Tensor, Tensor]]
    p_init: Sampleable[Tensor]
    f_model: FlowModel
    f_path: FlowPath
    vae: AutoEncoder

    def __post_init__(self) -> None:
        self.optim = AdamW(self.f_model.parameters(), lr=self.lr)
        self.ode = ODE(self.f_model)

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

            t = torch.rand((self.batch_size, 1), device=self.device)
            x_0 = self.p_init.sample(self.batch_size).flatten(1).to(self.device)
            x_1 = self.vae.encode(x).flatten(1)
            x_t = self.f_path.flow(x_0, x_1, t)

            v_true = self.f_path.cond_vector_field(x_1=x_1, x_t=x_t, t=t)
            v_pred = self.f_model.forward(x=x_t, t=t)

            loss = mse_loss(v_pred, v_true)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            if step % self.logger.step_interval == 0:
                pbar.set_description(f"{loss.item() = :.2f}")

                bs = self.logger.batch_size
                t = torch.linspace(0, 1, self.logger.sample_steps, device=self.device)
                x_0 = self.p_init.sample(bs).flatten(1).to(self.device)
                x_1 = self.ode.sample(x_0, t)
                x_1 = x_1.reshape((-1, 4, 4, 4))
                x_1 = self.vae.decode(x=x_1)
                x_1 = make_grid(x_1.cpu(), nrow=bs)

                f = plot_batch(x_1, x_1.size(0), show=False)
                mlflow.log_figure(f, "ode_sample_{}.png".format(step))
            mlflow.log_metric("train_loss", value=loss.item(), step=step)

        self.f_model.eval()
        pbar.close()
