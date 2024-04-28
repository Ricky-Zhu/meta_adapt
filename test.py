import torch
from torch.utils.data import DataLoader, Dataset

x = torch.rand((10, 8))
y = torch.rand((10, 1))

dl = DataLoader(Dataset(x, y), batch_size=3)
