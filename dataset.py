import keras
import torch
from torch.utils.data import TensorDataset
from keras import ops
import numpy as np

class HistoryDataset(TensorDataset):

    def __init__(self, x1, y, last_dim_mean = None, last_dim_std= None, keras = False):
    
        super().__init__()
        self.x1_mean, self.x1_std = x1.mean(axis=0), x1.std(axis=0)
        self.y_mean, self.y_std = y.mean(axis=0), y.std(axis=0)

        if last_dim_mean is not None and last_dim_std is not None:
            self.y_mean[-1] = last_dim_mean
            self.y_std[-1] = last_dim_std

        #if self.x_std or self.y_std == 0 on any axis, we set it to 1 to avoid division by zero
        self.x1_std[self.x1_std == 0] = 1
        if y.ndim == 1:
            self.y_std = 1 if self.y_std == 0 else self.y_std
        else :
            self.y_std[self.y_std == 0] = 1

        self.x1 = (x1 - self.x1_mean) / self.x1_std
        self.y = (y - self.y_mean) / self.y_std


        if keras == True:
            self.x1 = ops.convert_to_tensor(self.x1.astype(np.float32))
            self.y = ops.convert_to_tensor(self.y.astype(np.float32))
        
        else:
            self.x1 = torch.tensor(self.x1, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.y[idx]

class MyDataGenerator(keras.utils.Sequence):
    def __init__(self, x1, x2, last_dim_mean, last_dim_std, batch_size, model, sampling_batch_size=1024, **kwargs):
        # Initialize the data generator
        super().__init__(**kwargs) 
        self.x1, self.x2 = x1, x2
        self.last_dim_mean = last_dim_mean
        self.last_dim_std = last_dim_std
        self.batch_size = batch_size
        self.model = model
        self.sampling_batch_size = sampling_batch_size
        self.num_samples = x1.shape[0]

        self.normalize()

        self.x1 = ops.convert_to_tensor(self.x1.astype(np.float32))
        self.x2 = ops.convert_to_tensor(self.x2.astype(np.float32))

        # Generate the initial set of y targets
        #self.on_epoch_end()

    def normalize(self):
        self.x1_mean, self.x1_std = self.x1.mean(axis=0), self.x1.std(axis=0)
        self.x2_mean, self.x2_std = self.x2.mean(axis=0), self.x2.std(axis=0)
        self.x2_mean[-1], self.x2_std[-1] = self.last_dim_mean, self.last_dim_std

        # Avoid division by zero
        self.x1_std[self.x1_std == 0] = 1
        self.x2_std[self.x2_std == 0] = 1

        self.x1 = (self.x1 - self.x1_mean) / self.x1_std
        self.x2 = (self.x2 - self.x2_mean) / self.x2_std

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        
        # Get the batch of x1 and the corresponding pre-generated y
        batch_x1 = self.x1[start_index:end_index]
        batch_y = self.y[start_index:end_index]
        
        # The model's input is x1, and its target is the generated y
        return batch_x1, batch_y

    def on_epoch_end(self):
        print("\n--- Regenerating y targets for the next epoch... ---")
        print(f'sampling_batch_size: {self.sampling_batch_size}, num_samples: {self.num_samples}')
        # Here, we regenerate the entire 'y' dataset using 'x2'.
        y_batches = [] # Use a list to collect batches
        for i in range(0, self.num_samples, self.sampling_batch_size):
            print(f'iteration {1+ (i // self.sampling_batch_size)} of {self.num_samples // self.sampling_batch_size}')
            x2_batch = self.x2[i:i + self.sampling_batch_size]
            y_batch = self.model.sample(x2_batch)
            y_batches.append(y_batch)

        self.y = keras.ops.concatenate(y_batches, axis=0)
              
        
        # You could also shuffle your data here if you wanted
        p = np.random.permutation(len(self.x1))
        self.x1, self.x2, self.y = self.x1[p], self.x2[p], self.y[p]