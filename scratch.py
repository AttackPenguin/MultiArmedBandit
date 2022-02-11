import os

import numpy as np
import torch
from torch import nn


x = torch.rand((2, 11))
x = torch.reshape(x, [2, 1, 11])
print(x.shape)
