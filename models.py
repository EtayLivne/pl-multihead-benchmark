from typing import Dict
from lightning import LightningModule
import torch
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.aggregation import RunningMean
from torchvision.models import resnet50
from torch import split_copy, split

class OneHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_features=128*128, out_features=128*32, bias=False)
        self.layer2 =  nn.Linear(in_features=128*32, out_features=128*8, bias=False)
        self.layer3 = nn.Linear(in_features=128*8, out_features=128*2, bias=False)
        self.layer4 = nn.Linear(in_features=128*2, out_features=64*2, bias=False)
        self.layer5 = nn.Linear(in_features=64*2, out_features=32*2, bias=False)
        
        # The single head
        self.layer6 = nn.Linear(in_features=32*2, out_features=16*2, bias=False)
        self.layer7 = nn.Linear(in_features=16*2, out_features=8*2, bias=False)
        self.layer8 = nn.Linear(in_features=8*2, out_features=4*2, bias=False)
        self.layer9 = nn.Linear(in_features=4*2, out_features=2*2, bias=False)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)
        l9 = self.layer9(l8)
        l9_sig = self.activation(l9)
        
        return l9_sig

class TwoHeads(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_features=128*128, out_features=128*32, bias=False)
        self.layer2 =  nn.Linear(in_features=128*32, out_features=128*8, bias=False)
        self.layer3 = nn.Linear(in_features=128*8, out_features=128*2, bias=False)
        self.layer4 = nn.Linear(in_features=128*2, out_features=64*2, bias=False)
        self.layer5 = nn.Linear(in_features=64*2, out_features=32*2, bias=False)
        self.layer6 = nn.Linear(in_features=32*2, out_features=16*2, bias=False)
        self.layer7 = nn.Linear(in_features=16*2, out_features=8*2, bias=False)
        self.layer8 = nn.Linear(in_features=8*2, out_features=4*2, bias=False)

        self.layer9a =nn.Linear(in_features=4*2, out_features=2, bias=False)
        self.layer9b =nn.Linear(in_features=4*2, out_features=2, bias=False)
        self.activation_a = nn.Sigmoid()
        self.activation_b = nn.Sigmoid()
        
    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)

        l9a = self.layer9a(l8)
        l9a_sig = self.activation_a(l9a)
        l9b = self.layer9b(l8)
        l9b_sig = self.activation_b(l9b)
        
        return l9a_sig, l9b_sig


    
class FourHeads(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_features=128*128, out_features=128*32, bias=False)
        self.layer2 =  nn.Linear(in_features=128*32, out_features=128*8, bias=False)
        self.layer3 = nn.Linear(in_features=128*8, out_features=128*2, bias=False)
        self.layer4 = nn.Linear(in_features=128*2, out_features=64*2, bias=False)
        self.layer5 = nn.Linear(in_features=64*2, out_features=32*2, bias=False)
        self.layer6 = nn.Linear(in_features=32*2, out_features=16*2, bias=False)
        self.layer7 = nn.Linear(in_features=16*2, out_features=8*2, bias=False)
        self.layer8 = nn.Linear(in_features=8*2, out_features=4*2, bias=False)

        self.layer9a =nn.Linear(in_features=4*2, out_features=1, bias=False)
        self.layer9b =nn.Linear(in_features=4*2, out_features=1, bias=False)
        self.layer9c =nn.Linear(in_features=4*2, out_features=1, bias=False)
        self.layer9d =nn.Linear(in_features=4*2, out_features=1, bias=False)
        
        self.activation_a = nn.Sigmoid()
        self.activation_b = nn.Sigmoid()
        self.activation_c = nn.Sigmoid()
        self.activation_d = nn.Sigmoid()
        
    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)

        l9a = self.layer9a(l8)
        l9a_sig = self.activation_a(l9a)
        l9b = self.layer9b(l8)
        l9b_sig = self.activation_b(l9b)
        l9c = self.layer9c(l8)
        l9c_sig = self.activation_c(l9c)
        l9d = self.layer9d(l8)
        l9d_sig = self.activation_d(l9d)
        
        return l9a_sig, l9b_sig, l9c_sig, l9d_sig
