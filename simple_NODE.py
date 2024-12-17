import pymatreader
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt

import pickle
import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

import time
import logging
import statistics
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchdiffeq

import rich


class _ODEFunc(nn.Module):
    def __init__(self, module, autonomous=True):
        super().__init__()
        self.module = module
        self.autonomous = autonomous

    def forward(self, t, x):
        if not self.autonomous:
            x = torch.cat([torch.ones_like(x[:, [0]]) * t, x], 1)
        return self.module(x)


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 rtol: float = 1e-4, atol: float = 1e-4, adjoint: bool = True,
                 autonomous: bool = True):
        super().__init__()
        self.odefunc = _ODEFunc(odefunc, autonomous=autonomous)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.integration_time = torch.tensor([0, 1], dtype=torch.float32)

    @property
    def ode_method(self):
        return torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint

    def forward(self, x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        integration_time = integration_time.to(x.device)
        ode_method =  torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        out = ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        return out
    

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, t_span:torch.Tensor, learning_rate:float=5e-3):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task = 'multiclass', num_classes=2)
        self.predictions = []
        self.true_labels = []

    def forward(self, x):
        return self.model(x)

    def inference(self, x, time_span):
        return self.model(x, adjoint=False, integration_time=time_span)

    def inference_no_projection(self, x, time_span):
        return self.model.forward_no_projection(x, adjoint=False, integration_time=time_span)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred[-1]  # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred[-1]  # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        acc = self.accuracy(y_pred.softmax(dim=-1), y)
        self.log('val_accuracy', acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred[-1]  # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        acc = self.accuracy(y_pred.softmax(dim=-1), y)
        self.log('test_accuracy', acc, prog_bar=True, logger=True)
        self.predictions.append(y_pred)
        self.true_labels.append(y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def plot_results(self):
        predictions = torch.cat(self.predictions).cpu().detach().numpy()
        true_labels = torch.cat(self.true_labels).cpu().detach().numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(true_labels)), true_labels, label='True Labels', alpha=0.6)
        plt.scatter(range(len(predictions)), predictions.argmax(axis=0), label='Predictions', alpha=0.6)
        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.title('True Labels vs Predictions')
        plt.savefig('temptest')
    

class Learner2(pl.LightningModule):
    def __init__(self, model: nn.Module, t_span: torch.Tensor, learning_rate: float = 5e-3):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()  # Use MSE loss for time series prediction
        self.predictions = []
        self.true_values = []  # Store true time series values

    def forward(self, x):
        return self.model(x)

    def inference(self, x, time_span):
        return self.model(x, adjoint=False, integration_time=time_span)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)  # Predict time series values
        y_pred = y_pred[-1]  # Select the last time step of the predicted sequence
        loss = self.mse_loss(y_pred, y)  # Use MSE for regression
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred[-1]  # Select the last time step of the predicted sequence
        loss = self.mse_loss(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred[-1]
        loss = self.mse_loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.predictions.append(y_pred)
        self.true_values.append(y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def plot_results(self):
        predictions = torch.cat(self.predictions).cpu().detach().numpy()
        true_values = torch.cat(self.true_values).cpu().detach().numpy()

        # Plot predictions and true values over time
        plt.figure(figsize=(10, 5))
        plt.plot(true_values, label='True Values', alpha=0.6)
        plt.plot(predictions, label='Predictions', alpha=0.6)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('True Values vs Predictions')
        plt.savefig('temptest')
