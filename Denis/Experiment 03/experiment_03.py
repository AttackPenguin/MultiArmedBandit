import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from Denis.nn_models import MABInceptionModel2
from Denis.reward_generators import RewardGeneratorChallenging
from Denis.training_methods import train_track_loss

model = MABInceptionModel2(
    use_dropout=True,
    use_batch_norm=True
)

dttm_start = pd.Timestamp.now()
print(f"Started at: {dttm_start}")

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)
train_track_loss(
    model,
    loss_fn,
    optimizer,
    RewardGeneratorChallenging,
    batch_size=1024,
    save_dir=f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    save_interval=25
)
