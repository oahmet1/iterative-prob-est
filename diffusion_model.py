'''
Adapted from https://github.com/ShenghaoWu/Counterfactual-Generative-Models

'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import logging
import time
import matplotlib as mpl
from matplotlib import colors
import os
from scipy.special import softmax
import pickle
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
import numpy as np
import math
from torch.utils.data import DataLoader
from dataset import HistoryDataset

#Schedules
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class MLP(nn.Module):
    def __init__(self, cov_dimension, hidden_dim=128, num_blocks=4):
        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.cov_dimension = cov_dimension

        self.initial_dim = cov_dimension + 1 + 1
        # --- Embeddings ---
        self.network = nn.Sequential(
            nn.Linear(self.initial_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, cond, t):

        x = torch.cat((t, x, cond), dim=1)
        x = self.network(x)
        return x
    
class DDPM(nn.Module):
    def __init__(self, cov_dim, hidden_dim, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = MLP(cov_dim,hidden_dim).to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()
        
        # Register normalization parameters as buffers so they're saved with the model
        self.register_buffer('x_mean', torch.zeros(cov_dim))
        self.register_buffer('x_std', torch.ones(cov_dim))
        self.register_buffer('y_mean', torch.tensor(0.0))
        self.register_buffer('y_std', torch.tensor(1.0))

    def forward(self, x, cond):
        """
        this method is used in training, so samples t and noise randomly
        """
        batch_size = x.shape[0]
        _ts = torch.randint(1, self.n_T+1, (batch_size,1)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts] * x
            + self.sqrtmab[_ts] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, cond,  _ts / self.n_T)

        return self.loss_mse(noise, pred)

    def sample(self, cov: torch.Tensor, device):
        """
        Sample once per condition, assuming a 1-D output (feature_dim=1).

        Args:
            cov (torch.Tensor): batch of context/covariate vectors, shape (batch_size, context_dim).
            treat (torch.Tensor): batch of treatment indicators or features, shape (batch_size, treat_dim).
            device (str or torch.device).
            guide_w (float): classifier-free guidance weight.

        Returns:
            torch.Tensor: sampled outputs of shape (batch_size, 1).
        """
        batch_size = cov.shape[0]

        # Start from x_T ~ N(0, 1), shape (batch_size, 1)
        x_i = torch.randn(batch_size, 1, device=device)

        cov = cov.to(device)

        for t in tqdm(range(self.n_T, 0, -1), desc="DDPM Sampling", leave=False):

            # Build timestep tensor: shape (batch_size, 1), then duplicate
            t_lin = torch.full((batch_size, 1), float(t) / self.n_T, device=device)
            z = torch.randn(batch_size, 1, device=device)
    
            eps = self.nn_model(x_i, cov, t_lin)

            # Compute x_{t-1} = 1/√α_t * (x_t – ((1−α_t)/√(1−ᾱ_t)) * eps_combined) + √β_t * z
            coef1 = self.oneover_sqrta[t]
            coef2 = self.mab_over_sqrtmab[t]
            coef3 = self.sqrt_beta_t[t]

            x_i = coef1 * (x_i - coef2 * eps) + coef3 * z

        return x_i

    #train the diffusion model
    def fit(self,args, X, y, prev = None ):
        
        print(f'Training {args.model_name}')
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        n_epoch = args.n_epoch
        lrate = args.lr
        save_model = True
        save_dir = 'models/'
        os.makedirs(save_dir,  exist_ok = True) 

        if prev is not None:
            ds = HistoryDataset(X, y, last_dim_mean = prev.x_mean[-1], last_dim_std = prev.x_std[-1])
        else :
            ds = HistoryDataset(X, y)
        
        # Update the registered buffers with the normalization parameters
        self.x_mean.copy_(torch.tensor(ds.x_mean, dtype=torch.float32))
        self.x_std.copy_(torch.tensor(ds.x_std, dtype=torch.float32))
        self.y_mean.copy_(torch.tensor(ds.y_mean, dtype=torch.float32))
        self.y_std.copy_(torch.tensor(ds.y_std, dtype=torch.float32))

        training_batched = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=5)

        optim = torch.optim.Adam(self.parameters(), lr=lrate)

        epoch_pbar = tqdm(range(n_epoch), desc="Epochs")

        for ep in epoch_pbar:
            self.train()
            
            # Linear lrate decay
            current_lr = lrate * (1 - ep / n_epoch)
            optim.param_groups[0]['lr'] = current_lr
                        
            batch_pbar = tqdm(training_batched, desc=f"Epoch {ep+1}/{n_epoch}", leave=False)
            
            for i, (X_batch, y_batch) in enumerate(batch_pbar):
                optim.zero_grad()
                X_batch = X_batch.to(device)

                # If a model is passed, sample from it, otherwise use the y_batch directly
                if prev is not None:
                    y_batch = prev.sample(X_batch, device).detach()
                else:
                    y_batch = y_batch.unsqueeze(1).to(device)

                loss = self(y_batch, X_batch)
                loss.backward()
            
                # Use set_postfix for the inner bar to show the loss
                batch_pbar.set_postfix(loss=f"{loss:.4f}")
                optim.step()
            
            postfix_dict = {'last_loss': f'{loss:.4f}', 'lr': f'{current_lr:.2e}'}
            
            if save_model:
                save_path = os.path.join(save_dir, f"{args.model_name}.pth")
                torch.save(self.state_dict(), save_path)
                postfix_dict['saved'] = '✅'

            epoch_pbar.set_postfix(postfix_dict)

        print(f"\nTraining finished. Model saved at {save_path}")

    def predict(self, X, sample_size=64):
        batch_size = 256
        # Convert buffer tensors to numpy for normalization
        x_mean_np = self.x_mean.cpu().numpy()
        x_std_np = self.x_std.cpu().numpy()
        
        X = (X - x_mean_np) / x_std_np
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        #X = X.unsqueeze(1)  # add batch dimension
        X = X.repeat(batch_size, 1)
        X = X.to(self.device)

        results = []
        num_batches = math.ceil(sample_size / batch_size)
        bar = tqdm(range(num_batches), desc="Sampling, ")
        for _ in bar:
            samples = self.sample(X, self.device).detach().cpu().numpy()
            results.append(samples)
        samples = np.concatenate(results, axis=0)

        return samples  # return as numpy array