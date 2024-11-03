from typing import Dict
from lightning import LightningModule
import torch
from torch import nn
from torchmetrics import Accuracy
from models import OneHead, TwoHeads, FourHeads

class OneHeadedClassifier(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        
        self.lr = lr

        self.loss_fn =  nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=4)
        self.model = OneHead()

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Dict[str, torch.Tensor]):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.accuracy(y_hat.round(), y)
        return loss


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr,
        )
        return optimizer
    
class OneHeadedClassifierWithTwoAccuracies(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        
        self.lr = lr

        self.loss_fn =  nn.CrossEntropyLoss()
        self.accuracy1 = Accuracy("multiclass", num_classes=4)
        self.accuracy2 = Accuracy("multiclass", num_classes=4)
        self.model = OneHead()

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Dict[str, torch.Tensor]):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.accuracy1(y_hat.round(), y)
        self.accuracy2(y_hat.round(), y)
        return loss


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr,
        )
        return optimizer
    
class TwoHeadedClassifier(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        self.loss_fn1 =  nn.CrossEntropyLoss()
        self.accuracy1 = Accuracy("multiclass", num_classes=2)
        self.loss_fn2 =  nn.CrossEntropyLoss()
        self.accuracy2 = Accuracy("multiclass", num_classes=2)
        self.model = TwoHeads()
        
    def forward(self, x):
        output1, output2 = self.model(x)
        return output1, output2

    def step(self, batch: Dict[str, torch.Tensor]):
        x, y = batch
        y_hat1, y_hat2 = self(x)
        
        loss1 = self.loss_fn1(y_hat1, y)
        self.accuracy1(y_hat1.round(), y)
        loss2 = self.loss_fn2(y_hat2, y)
        self.accuracy2(y_hat2.round(), y)
        
        return loss1, loss2

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss1, loss2 = self.step(batch)
        return (loss1 + loss2) / 2

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss1, loss2 = self.step(batch)
        return (loss1 + loss2) / 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer


class FourHeadedClassifier(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        self.loss_fn1 =  nn.BCELoss()
        self.accuracy1 = Accuracy("binary", num_classes=1)
        self.loss_fn2 =  nn.BCELoss()
        self.accuracy2 = Accuracy("binary", num_classes=1)
        self.loss_fn3 =  nn.BCELoss()
        self.accuracy3 = Accuracy("binary", num_classes=1)
        self.loss_fn4 =  nn.BCELoss()
        self.accuracy4 = Accuracy("binary", num_classes=1)
        self.model = FourHeads()
        
    def forward(self, x):
        output1, output2, output3, output4 = self.model(x)
        return output1, output2, output3, output4

    def step(self, batch: Dict[str, torch.Tensor]):
        x, y = batch
        y_hat1, y_hat2, y_hat3, y_hat4 = self(x)
        
        loss1 = self.loss_fn1(y_hat1, y)
        self.accuracy1(y_hat1.round(), y)
        loss2 = self.loss_fn2(y_hat2, y)
        self.accuracy2(y_hat2.round(), y)
        loss3 = self.loss_fn3(y_hat3, y)
        self.accuracy3(y_hat1.round(), y)
        loss4 = self.loss_fn4(y_hat4, y)
        self.accuracy4(y_hat4.round(), y)
        
        return loss1, loss2, loss3, loss4

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss1, loss2, loss3, loss4 = self.step(batch)
        return (loss1 + loss2 + loss3 + loss4) / 4

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss1, loss2, loss3, loss4 = self.step(batch)
        return (loss1 + loss2 + loss3 + loss4) / 4

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer