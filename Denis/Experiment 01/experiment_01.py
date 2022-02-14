import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from Denis.nn_models import MABInceptionModel
from Denis.reward_generators import RewardGeneratorTruncNorm
from Denis.training_methods import training_method_01

model = MABInceptionModel()

dttm_start = pd.Timestamp.now()
print(f"Started at: {dttm_start}")

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.1
)
training_method_01(
    model,
    loss_fn,
    optimizer,
    RewardGeneratorTruncNorm,
    batch_size=32,
    save_dir='batch_size_32_lr_0.1',
    save_interval=50
)
