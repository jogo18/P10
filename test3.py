import torch
import torch.nn as nn
import pytorch_lightning as pl
import pymatreader
import numpy as np

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchdyn.models import NeuralODE
from torchmetrics import Accuracy


class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, solver_type="dopri5"):
        super(ODELSTMCell, self).__init__()
        self.solver_type = solver_type
        self.fixed_step_solver = solver_type.startswith("fixed_")
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        # 1 hidden layer NODE
        self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        if not self.fixed_step_solver:
            self.node = NeuralODE(self.f_node, solver=solver_type)
        else:
            options = {
                "fixed_euler": self.euler,
                "fixed_heun": self.heun,
                "fixed_rk4": self.rk4,
            }
            if not solver_type in options.keys():
                raise ValueError("Unknown solver type '{:}'".format(solver_type))
            self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        if hx[0].size(1) != self.hidden_size or hx[1].size(1) != self.hidden_size:
            hx = (torch.zeros(input.size(0), self.hidden_size, device=input.device),
                  torch.zeros(input.size(0), self.hidden_size, device=input.device))
        new_h, new_c = self.lstm(input, hx)
        if self.fixed_step_solver:
            new_h = self.solve_fixed(new_h, ts)
        else:
            indices = torch.argsort(ts)
            batch_size = ts.size(0)
            device = input.device
            s_sort = ts[indices]
            s_sort = s_sort + torch.linspace(0, 1e-4, batch_size, device=device)
            # HACK: Make sure no two points are equal
            trajectory = self.node.trajectory(new_h, s_sort)
            new_h = trajectory[indices, torch.arange(batch_size, device=device)]

        return (new_h, new_c)

    def solve_fixed(self, x, ts):
        ts = ts.view(-1, 1)
        for i in range(3):  # 3 unfolds
            x = self.node(x, ts * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)

        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class ODELSTM(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature,
        return_sequences=True,
        solver_type="dopri5",
    ):
        super(ODELSTM, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = ODELSTMCell(in_features, hidden_size, solver_type=solver_type)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, timespans, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = [
            torch.zeros((batch_size, self.hidden_size), device=device),
            torch.zeros((batch_size, self.hidden_size), device=device),
        ]
        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature), device=device)
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            if mask is not None:
                cur_mask = mask[:, t].view(batch_size, 1)
                last_output = cur_mask * current_output + (1.0 - cur_mask) * last_output
            else:
                last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # return entire sequence
        else:
            outputs = last_output  # only last item
        return outputs


class IrregularSequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, t, y, mask = batch
        else:
            x, t, y = batch
            mask = None
        y_hat = self.model.forward(x, t, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = Accuracy(preds, y)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, t, y, mask = batch
        else:
            x, t, y = batch
            mask = None
        y_hat = self.model.forward(x, t, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)

        loss = nn.CrossEntropyLoss()(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = Accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    



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

def main():
    seq_len = 10
    batch_size = 1

    model = ODELSTM(
    in_features = 1,
    hidden_size = 64,
    out_feature = 1
    ) # Modify the input size to match the expected size of the ODELSTM model
    data_module = CGM_DataModule(seq_len, batch_size)
    model = IrregularSequenceLearner(model)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()