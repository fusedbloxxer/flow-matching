from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import torch

from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split

from .config import Config
from .data import FlowersDataset, get_transform
from .model import AutoEncoder, DiCo
from .prob import OTProbPath, ProbPath
from .utils import set_manual_seed


@dataclass(kw_only=True)
class Trainer:
    u_theta: Module
    vae: AutoEncoder
    dataset: Dataset
    prob_path: ProbPath
    device: torch.device
    ckpt_dir: Path

    def __post_init__(self) -> None:
        self.dataset_size = len(cast(Any, self.dataset))

    def train_step(self) -> None:
        pass

    def train(
        self,
        *,
        bs: int,
        lr: float,
        seed: int,
        run_name: str,
        ckpt_name: str,
        ckpt_every: int,
        ckpt_resume: bool,
        eval_every: int,
        eval_split: float,
        steps: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> None:
        if steps is None and epochs is None:
            raise ValueError("Either steps or epochs should be specified")
        if steps is not None and epochs is not None:
            raise ValueError("Only one of steps or epochs should be specified")

        set_manual_seed(seed)
        train_dl, eval_dl = self.create_loaders(eval_split, batch_size=bs)

        if epochs:
            steps = epochs * len(train_dl)
        if not steps:
            raise ValueError("Number of steps could not be computed")

        for epoch in range(steps):
            pass

    @torch.no_grad()
    def eval(self) -> None:
        pass

    def save_ckpt(self) -> None:
        pass

    def load_ckpt(self) -> None:
        pass

    def create_optimizer(self) -> None:
        pass

    def create_loaders(self, eval_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        eval_ds, train_ds = random_split(self.dataset, lengths=[eval_split, 1.0 - eval_split])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        return train_dl, eval_dl


def train(cfg: Config):
    print(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    preprocess = get_transform(**asdict(cfg.data.preprocess))
    dataset = FlowersDataset(path=cfg.data.path, transform=preprocess)

    # Initialize models
    vae = AutoEncoder(**asdict(cfg.model.autoencoder))
    u_theta = DiCo(**asdict(cfg.model.vector_field))

    # Initialize CondOT
    prob_path = OTProbPath()

    # Initialize Trainer
    trainer = Trainer(
        ckpt_dir=cfg.train.ckpt.dir,
        prob_path=prob_path,
        dataset=dataset,
        device=device,
        u_theta=u_theta,
        vae=vae,
    )

    # Launch job
    trainer.train(
        seed=cfg.base.seed,
        lr=cfg.train.params.lr,
        bs=cfg.train.params.batch_size,
        steps=cfg.train.params.steps,
        epochs=cfg.train.params.epochs,
        run_name=cfg.track.run.name,
        ckpt_name=cfg.train.ckpt.name,
        ckpt_every=cfg.train.ckpt.every,
        eval_split=cfg.train.eval.split,
        eval_every=cfg.train.eval.every,
        ckpt_resume=cfg.train.ckpt.resume,
    )
