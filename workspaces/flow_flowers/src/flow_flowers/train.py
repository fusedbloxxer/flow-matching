from dataclasses import asdict, dataclass

import torch

from torch.nn import Module
from torch.utils.data import Dataset

from .config import Config, TrackConfig, TrainCkptConfig, TrainParamsConfig
from .data import FlowersDataset, get_transform
from .model import AutoEncoder, DiCo
from .prob import OTProbPath, ProbPath


@dataclass(kw_only=True)
class Trainer:
    u_theta: Module
    vae: AutoEncoder
    dataset: Dataset
    prob_path: ProbPath
    ckpt_config: TrainCkptConfig

    def __post_init__(self) -> None:
        pass

    def train_step(self) -> None:
        pass

    def train(self, *, params: TrainParamsConfig, track: TrackConfig) -> None:
        assert params.epochs, "Number of epochs must be specified"

        # dataloader = DataLoader(self.dataset, batch_size=params.batch_size, shuffle=True)

        for epoch in range(params.epochs):
            pass

    @torch.no_grad()
    def eval(self) -> None:
        pass

    def save_ckpt(self) -> None:
        pass

    def load_ckpt(self) -> None:
        pass


def train(cfg: Config):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create models
    vae = AutoEncoder(**asdict(cfg.model.autoencoder))
    u_theta = DiCo(**asdict(cfg.model.vector_field))

    # Create dataset
    preprocess = get_transform(**asdict(cfg.data.preprocess))
    dataset = FlowersDataset(path=cfg.data.path, transform=preprocess)

    # Optimal Transport Path
    prob_path = OTProbPath()

    # Create Trainer
    trainer = Trainer(
        ckpt_config=cfg.train.ckpt,
        prob_path=prob_path,
        dataset=dataset,
        u_theta=u_theta,
        vae=vae,
    )

    # Train
    trainer.train(params=cfg.train.params, track=cfg.track)
