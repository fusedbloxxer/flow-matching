import torchvision.transforms.v2 as T

from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
