from with_lightning.litmodules import OneHeadedClassifier, TwoHeadedClassifier, FourHeadedClassifier, OneHeadedClassifierWithTwoAccuracies
from lightning.pytorch import Trainer, LightningDataModule
from time import time
from dataset import VectorDataset
from torch.utils.data import DataLoader


# class VectorDatamodule(LightningDataModule):
#     def __init__(self, dataloader: DataLoader):
#         self._train_dataloader = dataloader
    
#     def train_dataloader(self):
#         return self._train_dataloader


def train(litmodule, train_dataset, num_steps, model_name):
    trainer = Trainer(
        max_steps=num_steps,
        barebones=True,
        devices=1
    )
    
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    start = time()    
    trainer.fit(
        model=litmodule,
        train_dataloaders=dataloader,
        
    )
    end = time()
    print(f"\n\n --------> {model_name}: {num_steps} in {end - start} seconds \n\n\n\n")
    

def train_one_head():
    m = OneHeadedClassifier()
    ds = VectorDataset(label_size=4)
    train(m, ds, 10_000, "ONE HEAD")
    
def train_one_head_with_two_accuracies():
    m = OneHeadedClassifierWithTwoAccuracies()
    ds = VectorDataset(label_size=4)
    train(m, ds, 10_000, "ONE HEAD TWO METRICS")
    
def train_two_heads():
    m = TwoHeadedClassifier()
    ds = VectorDataset(label_size=2)
    train(m, ds, 10_000, "TWO HEADS")

def train_four_heads():
    m = FourHeadedClassifier()
    ds = VectorDataset(label_size=1)
    train(m, ds, 10_000, "FOUR HEADS")