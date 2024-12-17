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

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



datapath = '/home/jagrole/AAU/9.Sem/Data/Processed_data_ALL.mat'

test = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')

# print(test['pkf']['timeSteps'][3])

CGM_data = test['pkf']['cgm']
timedata = test['pkf']['timecgm']
dosesdata = test['pkf']['doses_matched_FA']
# npdata = np.array(data)

# for i in data:
#     print(i.shape)
print(CGM_data[0].shape)
# print(CGM_data[1].shape)
# print(CGM_data[2].shape)

tempdata = CGM_data[0][0:3000]
temptimedata = timedata[0][0:3000]

# tempdoses = dosesdata[4][0:5000]
# temptime2 = timedata[4][0:5000]
plot = plt.plot(temptimedata, tempdata, '.')


plt.savefig('test.png')


class CGM_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CGM_Datamodule(pl.LightningDataModule):
    def __init__(self, folder_path, batch_size=32):
        super().__init__()
        self.folder_path = folder_path
        # self.data_path = data_path
        self.batch_size = batch_size

    def prepare_data(self):
        # data = pymatreader.read_mat(self.data_path)
        pass

        return 
    def setup(self, stage=None):
        self.data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'rb') as f:
                self.data.extend(pickle.load(f))
        # self.data = pymatreader.read_mat(self.data_path)
        # self.data = self.data['pkf']['cgm']
        # self.timedata = self.data['pkf']['timecgm']

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)
    
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

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, t_span:torch.Tensor, learning_rate:float=5e-3):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(num_classes=2)

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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
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


    cgm_sample, cgm_times =  CGM_Datamodule()


    # orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    cgm_sample = torch.from_numpy(cgm_sample).float().to(device)
    cgm_times = torch.from_numpy(cgm_times).float().to(device)


    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionODELSTM(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0) -> z0 the initial state for ODE solver
            h = rec.initHidden().to(device)
            c = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h, c = rec.forward(obs, h, c, torch.Tensor([t-1,t]))
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts, method='rk4').permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            #a = torch.nn.utils.clip_grad_norm_(params, 2.0)
            #print(a)
            optimizer.step()
            loss_meter.update(loss.item())

            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.visualize:
        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h , c = rec.forward(obs, h, c, torch.Tensor([t-1,t]))
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            orig_ts = torch.from_numpy(orig_ts).float().to(device)

            # take first trajectory for visualization
            z0 = z0[0]

            ts_pos = np.linspace(0., 2. * np.pi, num=2000)
            ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
            ts_pos = torch.from_numpy(ts_pos).float().to(device)
            ts_neg = torch.from_numpy(ts_neg).float().to(device)

            zs_pos = odeint(func, z0, ts_pos, method='rk4')
            zs_neg = odeint(func, z0, ts_neg, method='rk4')

            xs_pos = dec(zs_pos)
            xs_neg = torch.flip(dec(zs_neg), dims=[0])

        xs_pos = xs_pos.cpu().numpy()
        xs_neg = xs_neg.cpu().numpy()
        orig_traj = orig_trajs[0].cpu().numpy()
        samp_traj = samp_trajs[0].cpu().numpy()

        plt.figure()
        plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                 'g', label='true trajectory')
        plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                 label='learned trajectory (t>0)')
        plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
                 label='learned trajectory (t<0)')
        plt.scatter(samp_traj[:, 0], samp_traj[
                    :, 1], label='sampled data', s=3)
        plt.legend()
        plt.savefig('./vis.png', dpi=500)
        print('Saved visualization figure at {}'.format('./vis.png'))




datapath = '/home/jagrole/AAU/9.Sem/Data/Processed_data_ALL.mat'

test = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')

# print(test['pkf']['timeSteps'][3])

CGM_data = test['pkf']['cgm']
timedata = test['pkf']['timecgm']
dosesdata = test['pkf']['doses_matched_FA']
# npdata = np.array(data)

# for i in data:
#     print(i.shape)
print(CGM_data[0].shape)
# print(CGM_data[1].shape)
# print(CGM_data[2].shape)

tempdata = CGM_data[0][0:3000]
temptimedata = timedata[0][0:3000]

# tempdoses = dosesdata[4][0:5000]
# temptime2 = timedata[4][0:5000]


plot = plt.plot(temptimedata, tempdata, '.')
plt.savefig('test.png')


