import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader, Dataset, IterableDataset
from lightning.pytorch import LightningDataModule


class VectorDataset(IterableDataset):
    def __init__(self, label_size: tuple):
        self.label_size = label_size
        self._generator = np.random.default_rng(seed=773)
        
    def __len__(self):
        return 1_000_000_000
        
    def __iter__(self):
        size = 128*128
        for _ in range(len(self)):
            vec = self._generator.random(size=(size)).astype(np.float32) 
            label = self._generator.integers(low=0, high=2,size=self.label_size).astype(np.float32) 
            yield vec, label