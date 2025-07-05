import torch

from .config import Config
from .model import DiCo


def train(cfg: Config):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Choose dims
    B, C, H, W, N = 1, 32, 16, 16, 99

    # Sample data
    t = torch.rand((B))
    y = torch.randint_like(t, N, dtype=torch.int32)
    x_t = torch.randn((B, C, H, W))

    # Create model
    model = DiCo(in_dim=32, h_dim=128, out_dim=32, h_size=16, w_size=16, mlp_layers=4, blocks=24, n_class=N)

    # Send data & model to GPU
    x_t = x_t.to(device)
    t = t.to(device)
    y = y.to(device)
    model.to(device)

    # Forward pass through the model
    output = model.forward(x_t, t=t, y=y)

    # Print the output shape to verify
    print(output.shape)
