import torch

from .config import Config
from .model import DiCo


def train(cfg: Config):
    # Choose dims
    B, C, H, W, N = 8, 3, 28, 28, 100

    # Sample data
    t = torch.rand((B))
    y = torch.randint_like(t, N, dtype=torch.int32)
    c = {"t": t, "y": y}
    x_t = torch.randn((B, C, H, W))

    # Create model
    model = DiCo(in_dim=3, h_dim=256, out_dim=3, h_size=28, w_size=28, mlp_layers=1, blocks=1, n_class=N)

    # Forward pass through the model
    output = model.forward(x_t, **c)

    # Print the output shape to verify
    print(output.shape)
