import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from dataclasses import dataclass

from .path import ProbPath


@dataclass(kw_only=True)
class Trainer:
    lr: float
    steps: int
    batch_size: int
    loss_every: int
    device: torch.device
    model: Module
    path: ProbPath

    def train(self) -> None:
        self.model.train()
        pbar = tqdm(range(self.steps))
        optim = AdamW(self.model.parameters(), lr=self.lr)

        for step in pbar:
            z = self.path.sample_data(self.batch_size).to(self.device)
            x = self.path.sample_init(self.batch_size).to(self.device)
            t = torch.rand((self.batch_size, 1)).to(self.device)
            x_t = self.path.sample_path(x, z, t)

            optim.zero_grad()
            loss = self.get_loss(x_t, x, z, t)
            loss.backward()
            optim.step()

            if step % self.loss_every == 0:
                pbar.set_description(f"Loss: {loss.item():.2f}, Step: {step}")
        self.model.eval()

    def get_loss(self, x_t: Tensor, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        u_true = self.path.conditional_vector_field(x_t, z, t)
        u_pred = self.model(x_t, t)
        return F.mse_loss(u_pred, u_true)
