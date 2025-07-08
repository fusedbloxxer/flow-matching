from collections import defaultdict
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from pprint import pp
from typing import Any, Dict, Optional, Tuple, TypedDict, cast

import einops
import mlflow
import torch
import torch.nn.functional as F

from box import Box
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .config import Config
from .data import FlowersDataset, get_transform
from .model import AutoEncoder, DiCo
from .prob import OTProbPath, ProbPath
from .utils import batch_op, iter_loop, set_manual_seed


class DeviceMap(TypedDict):
    u_theta: str
    vae: str


@dataclass(kw_only=True)
class Trainer:
    cfg: float
    u_theta: DiCo
    vae: AutoEncoder
    dataset: Dataset
    ckpt_dir: Path
    track_uri: str
    prob_path: ProbPath
    experiment_name: str
    run_params: Dict[str, Any]
    devices: InitVar[DeviceMap] = field(default=cast(Any, {}))

    def __post_init__(self, devices: DeviceMap) -> None:
        mlflow.set_tracking_uri(uri=self.track_uri)
        mlflow.set_experiment(experiment_name=self.experiment_name)
        self.dataset_size = len(cast(Any, self.dataset))
        self.device_map: DeviceMap = cast(DeviceMap, defaultdict(lambda: "cpu", devices))

    def train(
        self,
        *,
        bs: int,
        lr: float,
        bs_vae: int,
        run_name: str,
        ckpt_name: str,
        ckpt_every: int,
        ckpt_resume: bool,
        log_every: int,
        eval_every: int,
        eval_split: float,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        epochs: Optional[int] = None,
        run_id: Optional[str] = None,
        run_nest: Optional[bool] = False,
    ) -> None:
        if steps is None and epochs is None:
            raise ValueError("Either steps or epochs should be specified")
        if steps is not None and epochs is not None:
            raise ValueError("Only one of steps or epochs should be specified")
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)
        if ckpt_resume and not (self.ckpt_dir / f"{ckpt_name}.pt").exists():
            raise ValueError("Checkpoint file does not exist")
        if run_id is None and ckpt_resume:
            raise ValueError("Run ID should be specified when resuming from a checkpoint")
        if run_id is not None and not ckpt_resume:
            raise ValueError("Run ID should be specified only if resuming from a checkpoint")
        if run_nest and not ckpt_resume:
            raise ValueError("Run nesting should be specified only if resuming from a checkpoint")
        if run_nest and run_id is None:
            raise ValueError("Run ID should be specified when nesting a run")

        if seed:
            set_manual_seed(seed)
        if epochs:
            steps = epochs * (self.dataset_size // bs)
        if not steps:
            raise ValueError("Number of steps could not be computed")
        if ckpt_resume:
            assert run_id
            start_step = int(ckpt_name.split("_")[-1])
            self.load_ckpt(name=ckpt_name)
        if not ckpt_resume:
            start_step = 0

        # Save params for training loop
        self._bs_vae = bs_vae

        # Create data loaders
        train_dl, eval_dl = self.create_loaders(eval_split, batch_size=bs)
        self._train_dl = iter_loop(train_dl)
        self._eval_dl = eval_dl

        # Resume tracking run
        run_opts = Box()
        run_opts.nested = run_nest
        run_opts.run_name = run_name
        run_opts.run_id = run_id if not run_nest else None
        run_opts.parent_run_id = run_id if run_nest else None

        # Overwrite run params
        run_params = Box(self.run_params)
        run_params.batch_size_vae = bs_vae
        run_params.batch_size = bs
        run_params.steps = steps
        run_params.seed = seed
        run_params.lr = lr

        # Create optimizer for u_theta
        self._optimizer = self.create_optimizer(lr=lr)

        # Move models to GPU
        self.u_theta.to(self.device_map["u_theta"])
        self.vae.to(self.device_map["vae"])
        self.vae.requires_grad_(False)
        self.vae.eval()

        with mlflow.start_run(**run_opts.to_dict()):
            mlflow.log_params(run_params, synchronous=False)

            for step in tqdm(range(start_step, start_step + steps)):
                self.train_step(step)

                if (step + 1) % eval_every == 0:
                    self.eval(start_step)

                if (step + 1) % ckpt_every == 0:
                    self.save_ckpt(f"{ckpt_name}_step_{step + 1}")

    def train_step(self, step: int) -> None:
        # TODO: Add MLFlow logs
        self.u_theta.train()

        # Sample (x_1_latent, y) ~ p_data
        x_1, y = next(self._train_dl)
        x_1: Tensor = x_1.to(self.device_map["vae"])
        x_1_latent = batch_op(x_1, self._bs_vae, lambda x: self.vae.encode(x).to(self.device_map["u_theta"]))

        # Apply CFG using pad_idx with drop probability
        y: Tensor = y.to(self.device_map["u_theta"])
        if self.cfg > 0.0:
            drop_mask = torch.rand(y.size(0), device=self.device_map["u_theta"])
            drop_mask = drop_mask <= self.cfg
            y[drop_mask] = self.u_theta.y_embedder.pad_idx
        y = einops.rearrange(y, "b -> b 1 1 1")

        # Sample t ~ U[0, 1)
        t = torch.rand((x_1.shape[0], 1, 1, 1), device=self.device_map["u_theta"])

        # Sample x_0 ~ p_init
        x_0 = torch.randn_like(x_1_latent, device=self.device_map["u_theta"])

        # Sample x_t ~ p_t(*|x_1)
        x_t_latent = self.prob_path.prob_path_flow(x_0=x_0, x_1=x_1_latent, t=t)

        # Learn CFM objective
        v_true = self.prob_path.target(x_1=x_1_latent, x_0=x_0)
        v_pred = self.u_theta(x_t=x_t_latent, t=t, y=y)
        loss = F.mse_loss(v_pred, v_true)

        # Backprop
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Logging
        mlflow.log_metric(key="train_loss", value=loss.item(), step=step, synchronous=False)

    @torch.no_grad()
    def eval(self, step: int) -> None:
        self.u_theta.eval()

    def save_ckpt(self, name: str) -> None:
        state_dict = self.u_theta.state_dict()
        torch.save(state_dict, self.ckpt_dir / f"{name}.pt")

    def load_ckpt(self, name: str) -> Module:
        state_dict = torch.load(self.ckpt_dir / f"{name}.pt")
        self.u_theta.load_state_dict(state_dict)
        return self.u_theta

    def create_loaders(self, eval_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        eval_ds, train_ds = random_split(self.dataset, lengths=[eval_split, 1.0 - eval_split])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        return train_dl, eval_dl

    def create_optimizer(self, lr: float) -> Optimizer:
        return AdamW(self.u_theta.parameters(), lr=lr, weight_decay=0.0)


def train(cfg: Config):
    if cfg.base.debug:
        pp(asdict(cfg), width=1, sort_dicts=False)

    # Setup environment
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize dataset
    preprocess = get_transform(**asdict(cfg.data.preprocess))
    dataset = FlowersDataset(path=cfg.data.path, transform=preprocess)

    # Initialize models
    vae = AutoEncoder(**asdict(cfg.model.autoencoder))
    u_theta = DiCo(**asdict(cfg.model.vector_field))

    # Gather run parameters
    run_params = Box()
    run_params.seed = cfg.base.seed
    run_params.lr = cfg.train.params.lr
    run_params.batch_size = cfg.train.params.batch_size
    run_params.num_classes = cfg.model.vector_field.n_class
    run_params.mlp_layers = cfg.model.vector_field.mlp_layers
    run_params.dico_layers = cfg.model.vector_field.blocks
    run_params.hidden_size = cfg.model.vector_field.h_dim
    run_params.latent_dim = cfg.model.vector_field.in_dim
    run_params.latent_w = cfg.model.vector_field.w_size
    run_params.latent_h = cfg.model.vector_field.h_size
    run_params.vae_id = cfg.model.autoencoder.id
    run_params.vae_mu = cfg.model.autoencoder.mu
    run_params.vae_std = cfg.model.autoencoder.std
    run_params.augment = cfg.data.preprocess.augment
    run_params.crop_size = cfg.data.preprocess.size
    run_params.crop_type = cfg.data.preprocess.crop
    run_params.norm = cfg.data.preprocess.norm

    if cfg.model.ddt:
        run_params.ddt = cfg.model.ddt.active
        run_params.ddt_encoder_layers = cfg.model.ddt.encoder
        run_params.ddt_decoder_layers = cfg.model.ddt.decoder

    # Initialize Trainer
    trainer = Trainer(
        vae=vae,
        u_theta=u_theta,
        dataset=dataset,
        prob_path=OTProbPath(),
        cfg=cfg.train.params.cfg,
        ckpt_dir=cfg.train.ckpt.dir,
        track_uri=cfg.track.server.uri,
        run_params=run_params.to_dict(),
        experiment_name=cfg.track.run.experiment,
        devices={"u_theta": "cuda:0", "vae": "cuda:0"},
    )

    # Launch job
    trainer.train(
        seed=cfg.base.seed,
        lr=cfg.train.params.lr,
        bs=cfg.train.params.batch_size,
        bs_vae=cfg.train.params.vae_batch_size,
        steps=cfg.train.params.steps,
        epochs=cfg.train.params.epochs,
        eval_split=cfg.train.eval.split,
        eval_every=cfg.train.eval.every,
        ckpt_name=cfg.train.ckpt.name,
        ckpt_every=cfg.train.ckpt.every,
        ckpt_resume=cfg.train.ckpt.resume,
        log_every=cfg.track.run.log_every,
        run_id=cfg.track.run.id,
        run_name=cfg.track.run.name,
        run_nest=cfg.track.run.nest,
    )
