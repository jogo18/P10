import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pymatreader
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
import argparse
from torchdiffeq import odeint_adjoint as odeint



data = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')




