import torch
import torch.nn as nn
import pytorch_lightning as pl
import pymatreader
import numpy as np

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchdyn.models import NeuralODE
from torchmetrics import Accuracy

import os
import argparse
import logging
import time
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionODELSTM(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionODELSTM, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.func = func

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h, c, t):
        h = odeint(self.func, h, t, method='rk4')[-1]

        combined = torch.cat((x, h), dim=1)

        c_tilde = torch.sigmoid(self.i2h(combined))
        i = torch.sigmoid(self.i2h(combined))
        f = torch.sigmoid(self.i2h(combined))
        o = torch.sigmoid(self.i2h(combined))

        c = f * c + i * c_tilde 

        h = o * torch.tanh(c)

        out = self.h2o(h)
        
        return out, h, c


    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1):
    v1 = torch.exp(lv1)
    lstd1 = lv1 / 2.

    kl = - lstd1 + ((v1 + (mu1) ** 2.) / 2.) - .5
    return kl

class CGM_DataModule(pl.LightningDataModule):
    def __init__(self, seq_len, batch_size): #split_ratio=(0.7, 0.2, 0.1)):
        super().__init__()
        self.data_full = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')
        self.seq_len = seq_len
        self.batch_size = batch_size
        # self.split_ratio = split_ratio

    def prepare_data(self) -> None:
        self.CGM_data = self.data_full['pkf']['cgm']
        self.CGM_conc = np.concatenate(self.CGM_data)
        # self.CGM_tensor = torch.tensor(npcgm)
        self.timedata = self.data_full['pkf']['timecgm']
        self.time_conc = np.concatenate(self.timedata)

    def setup(self, stage=None):
        split_idx = 4223934
        cgm_train, cgm_val  = self.CGM_conc[:split_idx], self.CGM_conc[split_idx:]
        times_train, times_val = self.time_conc[:split_idx], self.time_conc[split_idx:]

        # Convert lists to sequences

        self.x_train, self.times_train, self.y_train = self.create_sequences(cgm_train, times_train)
        self.x_val, self.times_val, self.y_val = self.create_sequences(cgm_val, times_val)

    def create_sequences(self, values, times):
        X, t,  y = [], [], []
        for i in range(len(values) - self.seq_len):
            seq_values = values[i:i + self.seq_len]
            seq_times = times[i:i + self.seq_len]
            x_seq_cgm = torch.tensor(seq_values, dtype=torch.float32)
            t_seq = torch.tensor(seq_times, dtype=torch.float32)
            y_seq_cgm = torch.tensor(values[i + self.seq_len], dtype=torch.long)
            X.append(x_seq_cgm)
            t.append(t_seq)
            y.append(y_seq_cgm)
        return torch.stack(X), torch.stack(t), torch.stack(y)
    
    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.times_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.x_val, self.times_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     test_dataset = TensorDataset(self.test_X, self.test_y)
    #     return DataLoader(test_dataset, batch_size=self.batch_len)

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    latent_dim = 25 
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)


    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionODELSTM(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

