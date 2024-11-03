import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import OneHead, TwoHeads, FourHeads
from dataset import VectorDataset
from time import time
from torchmetrics import Accuracy
    

class AccumulatedAccuracy:
    def __init__(self):
        device = torch.device("cuda")
        self.epoch_acc = torch.tensor([0], device=device, dtype=torch.float32, requires_grad=False)
        self._num_batches_in_curr_epoch = torch.tensor([0], device=device, dtype=torch.float32, requires_grad=False)
    
    def get_epoch_acc(self):
        return self.epoch_acc / self._num_batches_in_curr_epoch
    
    def reset_epoch_acc(self):
        self.epoch_acc = 0
        
    def __call__(self, predictions, labels):
        predicted_classes = torch.round(predictions)
        correct = (predicted_classes == labels).sum().item()
        batch_accuracy = correct / labels.numel()
        self.epoch_acc += batch_accuracy
        
        return batch_accuracy
        
        
        
        
def train_one_head():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHead().to(device)
    criterion = nn.CrossEntropyLoss()  # Binary Cross-Entropy Loss for binary classification
    acc_acc = AccumulatedAccuracy()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Prepare the dataset and data loader
    dataset = VectorDataset(label_size=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


    # Training loop
    num_epochs = 1
    start = time()
    for epoch in range(num_epochs):
        for _, data_tup in zip(range(10_000), dataloader):
            # Forward pass
            inputs, labels = data_tup
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()  # (batch_size, 1) -> (batch_size,)
            loss = criterion(outputs, labels)
            acc = acc_acc(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_acc = acc_acc.get_epoch_acc()
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {epoch_acc:.4f}")
        acc_acc.reset_epoch_acc()

    end = time()
    print(f"\n\n --------> ONE HEAD: {num_epochs} in {end - start} seconds \n\n\n\n")
    

def train_two_heads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeads().to(device)
    criterion1 = nn.CrossEntropyLoss()  
    criterion2 = nn.CrossEntropyLoss()  
    acc_acc1 = AccumulatedAccuracy()
    acc_acc2 = AccumulatedAccuracy()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Prepare the dataset and data loader
    dataset = VectorDataset(label_size=2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


    # Training loop
    num_epochs = 1
    start = time()
    for epoch in range(num_epochs):
        for _, data_tup in zip(range(10_000), dataloader):
            # Forward pass
            inputs, labels = data_tup
            inputs, labels = inputs.to(device), labels.to(device)
            output1, output2 = model(inputs)
            output1 = output1.squeeze()
            output2 = output2.squeeze()
            loss1 = criterion1(output1, labels)
            acc1 = acc_acc1(output1, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            loss2 = criterion2(output2, labels)
            acc2 = acc_acc2(output2, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            # Backward pass and optimization
            loss = (loss1 + loss2) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_acc1 = acc_acc1.get_epoch_acc()
        epoch_acc2 = acc_acc2.get_epoch_acc()
        acc_acc1.reset_epoch_acc()
        acc_acc2.reset_epoch_acc()

    end = time()
    print(f"\n\n --------> TWO HEADS: {num_epochs} in {end - start} seconds \n\n\n\n")


def train_four_heads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourHeads().to(device)
    criterion1 = nn.BCELoss()  
    criterion2 = nn.BCELoss()
    criterion3 = nn.BCELoss()  
    criterion4 = nn.BCELoss()  
    acc_acc1 = AccumulatedAccuracy()
    acc_acc2 = AccumulatedAccuracy()
    acc_acc3 = AccumulatedAccuracy()
    acc_acc4 = AccumulatedAccuracy()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Prepare the dataset and data loader
    dataset = VectorDataset(label_size=1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


    # Training loop
    num_epochs = 1
    start = time()
    for epoch in range(num_epochs):
        for _, data_tup in zip(range(10_000), dataloader):
            # Forward pass
            inputs, labels = data_tup
            inputs, labels = inputs.to(device), labels.to(device)
            output1, output2, output3, output4 = model(inputs)
            output1 = output1.squeeze()
            output2 = output2.squeeze()
            output3 = output3.squeeze()
            output4 = output4.squeeze()
            labels = labels.squeeze()
            loss1 = criterion1(output1, labels)
            acc1 = acc_acc1(output1, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            loss2 = criterion2(output2, labels)
            acc2 = acc_acc2(output2, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            loss3 = criterion3(output3, labels)
            acc3 = acc_acc3(output3, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            loss4 = criterion4(output4, labels)
            acc4 = acc_acc4(output4, labels)    # same labels used for both losses cause I don't actually care about the data itself, just the shapes
            # Backward pass and optimization
            loss = (loss1 + loss2 + loss3 + loss4) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_acc1 = acc_acc1.get_epoch_acc()
        epoch_acc2 = acc_acc2.get_epoch_acc()
        epoch_acc3 = acc_acc3.get_epoch_acc()
        epoch_acc4 = acc_acc4.get_epoch_acc()
        acc_acc1.reset_epoch_acc()
        acc_acc2.reset_epoch_acc()
        acc_acc3.reset_epoch_acc()
        acc_acc4.reset_epoch_acc()


    end = time()
    print(f"\n\n --------> FOUR HEADS: {num_epochs} in {end - start} seconds \n\n\n\n")