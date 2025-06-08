import torch
import matplotlib.pyplot as plt

from torch import Tensor
from einops import reduce
from matplotlib.axes import Axes


def plot_prob_path(x_t: Tensor, t: Tensor) -> None:
    # t   - T
    # x_t - T x N x D
    n_steps  = x_t.size(0)
    _, axes = plt.subplots(nrows=1, ncols=n_steps, figsize=(20, 8))

    x_min, y_min = reduce(x_t, 't n d -> d', 'min')
    x_max, y_max = reduce(x_t, 't n d -> d', 'max')

    for t_step, axis in enumerate(axes):
        axis: Axes
        axis.scatter(x_t[t_step, :, 0], x_t[t_step, :, 1], marker=".")
        axis.set_xlim(x_min.item(), x_max.item())
        axis.set_ylim(y_min.item(), y_max.item())
        axis.locator_params("both", nbins=8)
        axis.set_title(f"t={t[t_step]:.2f}")
        axis.set_aspect("equal")
        axis.grid(True)
    plt.tight_layout()


def plot_trajectory(x_t: Tensor) -> None:
    num_instances = x_t.shape[1]

    plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    for i in range(num_instances):
        x = x_t[:, i, 0]
        y = x_t[:, i, 1]
        plt.plot(x, y, color='blue', linewidth=0.5) # Use a single color for all trajectories

    plt.title("Trajectories of Points Over Time")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
