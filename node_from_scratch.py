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
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':

    #DO NEXT: MAKE NEW CLASS THAT COMBINES ENCODER AND DECODER IN LIGHTNING MODULE THEN TEST IT SO THAT IT RUNS CORRECTLY
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
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    test_data = CGM_DataModule(seq_len=10, batch_size=1)
    test_data.prepare_data()
    test_data.setup()
    train_loader = test_data.train_dataloader()
    data_point = next(iter(train_loader))
    x, t, y = data_point
    print(x.shape, t.shape, y.shape)

    model = RecognitionODELSTM(latent_dim, obs_dim, nhidden, nbatch=1).to(device)
