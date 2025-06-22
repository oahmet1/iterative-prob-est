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
from torch.utils.data import Dataset, DataLoader

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

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, double_layers=False):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            #nn.Linear(emb_dim * 4, emb_dim * 2 ),
            #nn.GELU(),
            #nn.Linear(emb_dim * 2, emb_dim ),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # embedding values need to be small
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        #print(f"embeddings shape: {embeddings.shape}, time shape: {time.shape}")
        embeddings = embeddings * time[:, None]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLP(nn.Module):
    def __init__(self, cov_dimension, hidden_dim = 128 ):
        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.cov_dimension = cov_dimension

        self.time_pe = SinusoidalPositionEmbeddings(hidden_dim)  # fixed embeddings
        self.encode_out = EmbedFC(1, hidden_dim // 2)
        self.encode_cond = EmbedFC(cov_dimension, hidden_dim // 2)
        
        self.mid_layers =  nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim ), nn.GELU(),
                                      ])
        
        self.main_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim ), nn.GELU(),
                                      nn.Linear(hidden_dim , hidden_dim ),nn.GELU(),
                                      nn.Linear(hidden_dim , 1 )
                                      ])

    def forward(self, x, cond, cond_mask, t):

        # x is (noisy) data, c is covariate, t is timestep, 
        # context_mask says which samples to block the context on
        #print(f"Input shapes - x: {x.shape}")
        x = self.encode_out(x)

        #print(f"Input shapes - x: {x.shape}")

        cond = self.encode_cond(cond)


        # if the mask is 0, make it 
        cond_mask = 1 - cond_mask # need to flip 0 <-> 1

        cond = cond * cond_mask  # apply mask to covariates

        #print time shape
        #print(f"time shape: {t.shape}")
        t = self.time_pe(t)  # positional encoding for time
        #for layer in self.encode_time:
        #    t = layer(t)
        
        #print(f"time embedding shape: {t.shape}")

        # print shapes for debugging
        #print(f"NNMODEL x shape: {x.shape}, treat shape: {treat.shape}, cov shape: {cov.shape}, t shape: {t.shape}")

        x = torch.cat((x, cond), 1)

        for layer in self.mid_layers:
            x = layer(x)

        x = x.add(t)  # add time embedding

        for layer in self.main_layers:
            x = layer(x)
        return x

class DDPM(nn.Module):
    def __init__(self, cov_dim, hidden_dim, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = MLP(cov_dim,hidden_dim).to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, cond):
        """
        this method is used in training, so samples t and noise randomly
        """
        batch_size = x.shape[0]
        _ts = torch.randint(1, self.n_T+1, (batch_size,)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        p_tensor = torch.full((batch_size, 1), self.drop_prob, device=self.device)
        cond_mask = torch.bernoulli(p_tensor)  # shape = [batch_size, 1], values are 0/1

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, cond, cond_mask,  _ts / self.n_T)

        return self.loss_mse(noise, pred)

    def sample(self, cov: torch.Tensor, device, guide_w: float = 0.0):
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

        # Move cov and treat to device
        cov = cov.to(device)

        # Create masks for cov and treat (zeros = use true value, ones = mask/unconditional)
        cov_mask = torch.zeros((batch_size, 1), device=self.device)

        # Duplicate cov, treat, and their masks along the batch dimension
        cov = cov.repeat(2, 1)
        cov_mask = cov_mask.repeat(2, 1)

        # In the second half, zero out (i.e. mask) both cov and treat
        cov_mask[batch_size:] = 1.0

        #print(f"cov shape: {cov.shape}, treat shape: {treat.shape}, cov_mask shape: {cov_mask.shape}, treat_mask shape: {treat_mask.shape}")

        for t in range(self.n_T, 0, -1):
            # Build timestep tensor: shape (batch_size, 1), then duplicate
            t_lin = torch.full((batch_size, ), float(t) / self.n_T, device=device)
            t_is = t_lin.repeat(2)

            # Because we doubled cov and treat, replicate x_i as well
            x_i = x_i.repeat(2, 1)

            # Sample noise z_t for next step (zero at t=1)
            if t > 1:
                z = torch.randn(batch_size, 1, device=device)
            else:
                z = torch.zeros(batch_size, 1, device=device)

            #print shapes of xi cov treat cov_mask treat_mask t_is
            #print(f"t: {t}, x_i shape: {x_i.shape}, cov shape: {cov.shape}, treat shape: {treat.shape}, cov_mask shape: {cov_mask.shape}, treat_mask shape: {treat_mask.shape}, t_is shape: {t_is.shape}")
            
            # Predict εθ for both “with-cov/treat” and “without-cov/treat” (masked)
            eps = self.nn_model(x_i, cov, cov_mask, t_is)
            #print(f"t: {t}, eps shape: {eps.shape}")
            eps1 = eps[:batch_size]       # predictions when cov_mask & treat_mask = 0
            eps2 = eps[batch_size:]       # predictions when cov_mask & treat_mask = 1

            #print(f"t: {t}, eps1 shape: {eps1.shape}, eps2 shape: {eps2.shape}")
            # Combine via classifier-free guidance
            eps_combined = (1 + guide_w) * eps1 - guide_w * eps2

            #print(f"t: {t}, eps_combined shape: {eps_combined.shape}")

            # Keep only the first half of x_i to update
            x_prev = x_i[:batch_size]

            # Compute x_{t-1} = 1/√α_t * (x_t – ((1−α_t)/√(1−ᾱ_t)) * eps_combined) + √β_t * z
            coef1 = self.oneover_sqrta[t]
            coef2 = self.mab_over_sqrtmab[t]
            coef3 = self.sqrt_beta_t[t]

            #print(f"t: {t}, coef1: {coef1.shape}, coef2: {coef2.shape}, coef3: {coef3.shape}, eps_combined shape: {eps_combined.shape}, z shape: {z.shape} x_prev shape: {x_prev.shape}")
            x_i = coef1 * (x_prev - coef2 * eps_combined) + coef3 * z

        # Final tensor has shape (batch_size, 1)
        return x_i

    #train the diffusion model
    def fit(self,args, X, y):
        
        print(f'Training {args.model_name}')
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        n_T = args.n_T
        n_epoch = args.n_epoch
        lrate = args.lr
        save_model = True
        save_dir = 'models/'
        os.makedirs(save_dir,  exist_ok = True) 

        ds = HistoryDataset(X, y)
        self.x_mean, self.x_std = ds.x_mean, ds.x_std
        self.y_mean, self.y_std = ds.y_mean, ds.y_std

        training_batched = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=5)

        optim = torch.optim.Adam(self.parameters(), lr=lrate)

        epoch_pbar = tqdm(range(n_epoch), desc="Epochs")

        for ep in epoch_pbar:
            self.train()
            
            # Linear lrate decay
            current_lr = lrate * (1 - ep / n_epoch)
            optim.param_groups[0]['lr'] = current_lr
            
            loss_ema = None
            
            # --- KEY CHANGE 2: Configure the inner loop ---
            # `leave=False` makes this bar disappear after the epoch is done.
            # The description now clearly shows which epoch is running.
            batch_pbar = tqdm(training_batched, desc=f"Epoch {ep+1}/{n_epoch}", leave=False)
            
            for i, (X_batch, y_batch) in enumerate(batch_pbar):
                optim.zero_grad()
                X_batch = X_batch.to(device)
                y_batch = y_batch.unsqueeze(1).to(device)

                loss = self(y_batch, X_batch)
                loss.backward()
                
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                
                # Use set_postfix for the inner bar to show the live EMA loss
                batch_pbar.set_postfix(loss=f"{loss_ema:.4f}")
                optim.step()
            
            # --- KEY CHANGE 3: Remove prints and update the outer bar instead ---
            # Instead of printing, update the main epoch progress bar with the final info.
            postfix_dict = {'last_loss': f'{loss_ema:.4f}', 'lr': f'{current_lr:.2e}'}
            
            if save_model:
                save_path = os.path.join(save_dir, f"{args.model_name}.pth")
                torch.save(self.state_dict(), save_path)
                # Add a visual indicator that the model was saved
                postfix_dict['saved'] = '✅'

            epoch_pbar.set_postfix(postfix_dict)

        print(f"\nTraining finished. Model saved at {save_path}")

    def predict(self, X, sample_size=64, guide_w=0.0):
        X = (X - self.x_mean) / self.x_std
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        #X = X.unsqueeze(1)  # add batch dimension
        X = X.repeat(sample_size, 1)
        X = X.to(self.device)
        # Sample from the model
        samples = self.sample(X, self.device, guide_w = guide_w)  #
        # Rescale the samples back to original scale
        samples = samples * self.y_std + self.y_mean
        return samples.cpu().detach().numpy()  # return as numpy array


class HistoryDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0)
        self.y_mean, self.y_std = y.mean(axis=0), y.std(axis=0)

        self.X = (X - self.x_mean) / self.x_std
        self.y = (y - self.y_mean) / self.y_std

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]