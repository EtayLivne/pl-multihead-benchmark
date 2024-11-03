from just_torch.train import train_one_head, train_two_heads, train_four_heads
from lightning.pytorch import seed_everything


seed_everything(111)
train_four_heads()
train_two_heads()
train_one_head()
