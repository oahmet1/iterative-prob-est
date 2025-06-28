import keras
import torch
from torch.utils.data import TensorDataset
from keras import ops
import numpy as np

class HistoryDataset(TensorDataset):

    def __init__(self, X, y, last_dim_mean = None, last_dim_std= None, keras = False):
    
        super().__init__()
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0)
        self.y_mean, self.y_std = y.mean(axis=0), y.std(axis=0)

        if last_dim_mean is not None and last_dim_std is not None:
            self.y_mean[-1] = last_dim_mean
            self.y_std[-1] = last_dim_std

        #if self.x_std or self.y_std == 0 on any axis, we set it to 1 to avoid division by zero
        self.x_std[self.x_std == 0] = 1
        if y.ndim == 1:
            self.y_std = 1 if self.y_std == 0 else self.y_std
        else :
            self.y_std[self.y_std == 0] = 1

        self.X = (X - self.x_mean) / self.x_std
        self.y = (y - self.y_mean) / self.y_std


        if keras == True:
            self.X = ops.convert_to_tensor(self.X.astype(np.float32))
            self.y = ops.convert_to_tensor(self.y.astype(np.float32))
        
        else:
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]