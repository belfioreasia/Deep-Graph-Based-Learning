import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
import torch.nn.functional as F

def block(in_dim, out_dim, device):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, device=device),
        nn.LeakyReLU())

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.device = args['device']

        self.lr_dim = args['lr_dim']
        self.hr_dim = args['hr_dim']
        self.hidden_dim = args['hidden_dim']*2
        self.num_layers = args['num_layers']

        self.layers = nn.ModuleList([block(self.lr_dim*self.lr_dim, self.hidden_dim, device=self.device)])
        for _ in range(self.num_layers-2):
            self.layers.append(block(self.hidden_dim, self.hidden_dim, device=self.device))
        self.layers.append(nn.Linear(self.hidden_dim, self.hr_dim*self.hr_dim, device=self.device))

    def forward(self, lr_A):
        with torch.autograd.set_detect_anomaly(True, True):
            z = lr_A
            for layer in self.layers:
                # print(z.shape)
                z = layer(z)
        return z
    
    def run_epoch(self, data_loader, optim, device, train=True):
        """
        Run one epoch of training or evaluation.
        Args:
            model: Model to train or evaluate
            data_loader: DataLoader object
            optim: Optimizer object
            device: Device to run the model on
            train: Boolean to specify training or evaluation
        Returns:
            epoch_loss: Loss for the current epoch
        """
        self.train() if train else self.eval()
        batch_count, epoch_loss = 0, 0.0

        for A_lr, A_hr in data_loader:
            batch_count += 1
            if train:
                optim.zero_grad()
            A_lr = A_lr.to(device)
            A_hr = A_hr.to(device)
            pred_A_hr = self.forward(A_lr)
            # print(pred_A_hr.shape, A_hr.shape)
            loss = F.mse_loss(pred_A_hr, A_hr)
            if train:
                loss.backward()
                optim.step()
            epoch_loss += loss.item()

        epoch_loss /= batch_count
        return epoch_loss
    
    def train_model(self, optim, lr, n_epoch, train_loader, test_loader, device, patience=10):

        best_val_loss = float('inf')
        patience_counter = 0

        train_loss_history = []
        val_loss_history = []

        pbar = tqdm(range(n_epoch))
        for ep in pbar:
            # Train for one epoch
            optim.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)
            train_loss_epoch = self.run_epoch(train_loader, optim, device, train=True)
            train_loss_history.append(train_loss_epoch)

            # Evaluate on validation set
            with torch.no_grad():
                val_loss_epoch = self.run_epoch(test_loader, optim, device, train=False)
            val_loss_history.append(val_loss_epoch)

            pbar.set_description(f"Epoch {ep} - train_loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}")
            pbar.write(f"Epoch {ep}, train_loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}")

            # Early Stopping Check
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                patience_counter = 0
                # Optionally save the best model state
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    pbar.write(f"Early stopping triggered after {ep} epochs.")
                    break
        # if save:
        #     with torch.no_grad():
        #         with open(config["Diffusion"][f"load_dir_{fold+1}"], "wb") as f:
        #             torch.save(best_model_state, f)
        return train_loss_history, val_loss_history
    