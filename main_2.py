'''
Creating a minimal dataset with a history dependency.

Structure:
X_0, X_1, X_2 X_3, X_4
A_0, A_1, A_2, A_3, A_4
                         Y
'''
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from dataset import HistoryDataset, MyDataGenerator
from utils import plot_two_histograms_on_same, plot_histogram
from diffusion_model import DDPM
from edm import EDM, get_EDM
import argparse
import torch
import keras

def time_series_conditional_data(X, sample_size=1000, plot=False):
    A_0, A_1, A_2 = X[3], X[4], X[5]
    X_0, X_1, X_2 = X[0], X[1], X[2]

    X_3 = iterate_X(X_0, X_1, X_2, A_0, A_1, A_2)  
    A_3 = iterate_A(X_0, X_1, X_2, X_3, A_0, A_1, A_2)
    X_4 = iterate_X(X_1, X_2, X_3, A_1, A_2, A_3)
    A_4 = iterate_A(X_1, X_2, X_3, X_4, A_1, A_2, A_3)

    m = 100
    means = np.array([-m + A_2, A_3 + X_2 + X_3 + X_4, m + A_4])  # shape (3,)

    components = np.random.choice(3, size=sample_size)

    Y = np.array([np.random.normal(loc=means[comp], scale=2.0) for comp in components])
    if plot:
        plot_histogram(Y)
    return Y


def time_series_conditional_interventional_data(X, intervention, sample_size=1000, plot=False):
    A_0, A_1, A_2 = X[3], X[4], intervention[0]
    X_0, X_1, X_2 = X[0], X[1], X[2]

    A_3, A_4 = intervention[1], intervention[2]
    X_3 = iterate_X(X_0, X_1, X_2, A_0, A_1, A_2)
    X_4 = iterate_X(X_1, X_2, X_3, A_1, A_2, A_3)


    m = 100
    means = np.array([-m + A_2, A_3 + X_2 + X_3 + X_4, m + A_4])  # shape (3,)

    components = np.random.choice(3, size=sample_size)

    Y = np.array([np.random.normal(loc=means[comp], scale=2.0) for comp in components])
    if plot:
        plot_histogram(Y)
    return Y

def iterate_A(X_0, X_1, X_2, X_3, A_0, A_1, A_2):
    coeff = np.array([-2.0, 2.0, -1.0, -0.5, -1.0, 2.0, 2.0])  # Coefficients for X_0, X_1, X_2, A_0, A_1, A_2 in order for calculating X_3
    A_3 = coeff[0] * X_0 + coeff[1] * X_1 + coeff[2] * X_2 + coeff[3] * X_3 + coeff[4] * A_0 + coeff[5] * A_1 + coeff[6] * A_2
    return A_3

def iterate_X(X_0, X_1, X_2, A_0, A_1, A_2):
    coeff = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])  # Coefficients for X_0, X_1, X_2, A_0, A_1, A_2 in order for calculating X_3
    X_3 = coeff[0] * X_0 + coeff[1] * X_1 + coeff[2] * X_2 + coeff[3] * A_0 + coeff[4] * A_1 + coeff[5] * A_2
    return X_3

def time_series_data(size, plot=False):
    # Create the dataset with gaussian noise
    A_0, A_1, A_2 = (np.random.normal(loc=0.0, scale=5, size=size) for _ in range(3))
    X_0, X_1, X_2 = (np.random.normal(loc=0.0, scale=3.0, size=size) for _ in range(3))

    X_3 = iterate_X(X_0, X_1, X_2, A_0, A_1, A_2)
    A_3 = iterate_A(X_0, X_1, X_2, X_3, A_0, A_1, A_2)
    X_4 = iterate_X(X_1, X_2, X_3, A_1, A_2, A_3)
    A_4 = iterate_A(X_1, X_2, X_3, X_4, A_1, A_2, A_3)


    m = 100
    means = np.stack([-m + A_2, A_3 + X_2 + X_3 + X_4, m + A_4], axis=1)

    components = np.random.choice(3, size=size)

    Y = np.array([
    np.random.normal(loc=means[i, comp], scale=2.0) for i, comp in enumerate(components) ])

    cov_1 = np.column_stack((X_0, X_1, X_2, A_0, A_1, A_2))
    cov_2 = np.column_stack((X_1, X_2, X_3, A_1, A_2, A_3))
    cov_3 = np.column_stack((X_2, X_3, X_4, A_2, A_3, A_4))
    X = np.stack((cov_1, cov_2, cov_3), axis=0)
    if plot:
        plot_histogram(Y)

    return X, Y

def estimate_iteratively(X, Y):
    
    X_test = np.array([[-2., 3. , 2. , -3.0, 2.0, 0.0]]) #np.array([[0., -2. , 0. ,4.0, 2.0, 0.0]])
    intervention = np.array([-5.0, -3.0, 0.0]) #np.array([5.0, 5.0, -2.0])    

    args_dict = {
        'n_epoch': 64,
        'model_name': 'diff_mlp',
        'hidden_dim': 256,
        'batch_size': 256,
        'n_T': 2000,
        'lr': 1e-3
        }
   
    args = argparse.Namespace(**args_dict)
   
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    prev = DDPM(cov_dim=X[2].shape[1], hidden_dim=args.hidden_dim, betas=(1e-4, 0.02), device=device, n_T=args.n_T)
    prev.to(device)
    prev.fit(args, X[2], Y)
    y_mean, y_std = prev.y_mean, prev.y_std

    for i in range(2, 0, -1):
        print(f"Iteration {-i + 3}")
        intervened_data = X[i]
        intervened_data[:, -1] = intervention[i]
        curr = DDPM(cov_dim=X[2].shape[1], hidden_dim=args.hidden_dim, betas=(1e-4, 0.02), device=device, n_T=args.n_T)
        curr.to(device)
        curr.fit(args, X[i-1], intervened_data,  prev=prev)
        prev = curr

    print("Sampling from the final model")
    X_test_intervened = X_test.copy()
    X_test_intervened[:, -1] = intervention[0]
    num_samples = 1000
    samples = curr.predict(X_test_intervened, num_samples)
    samples = samples * y_std + y_mean  # Rescale the samples
    plot_two_histograms_on_same(time_series_conditional_interventional_data(X_test[0], intervention), samples)

def estimate_iteratively_edm(X, Y):
    print(f'shape of cov : {X[0].shape}')
    X_test_original = np.array([[-2., 3. , 2. , -3.0, 2.0, 0.0]]) #np.array([[0., -2. , 0. ,4.0, 2.0, 0.0]])
    intervention = np.array([-5.0, -3.0, 0.0]) #np.array([5.0, 5.0, -2.0])    

    args_dict = {
        'n_epoch': 25,
        'hidden_dim': 256,
        'n_hidden': 6,
        'batch_size': 512,
        'lr': 1e-3,
        'sampling_batch_size': 2 ** 17,
    }
    args = argparse.Namespace(**args_dict)

    Y = Y[:, None]
    ds_prev = HistoryDataset(X[2], Y, keras=True)
    data_len = ds_prev.__len__()
    y_mean, y_std = ds_prev.y_mean, ds_prev.y_std
    training_batched = torch.utils.data.DataLoader(ds_prev, batch_size=args.batch_size, shuffle=True )

    prev = get_EDM(cov_dim = X[2].shape[1], hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, 
                   data_len=data_len, num_epochs=args.n_epoch, lr=args.lr, batch_size=args.batch_size)
    prev.fit(training_batched, epochs=args.n_epoch, verbose=1)

    for i in range(2, 0, -1):
        print(f"Iteration {-i + 3}")
        
        intervened_data = X[i].copy()
        intervened_data[:, -1] = intervention[i]
        ds_curr = MyDataGenerator(X[i-1], intervened_data, last_dim_mean=ds_prev.x1_mean[-1], last_dim_std=ds_prev.x1_std[-1], batch_size=args.batch_size, model=prev, sampling_batch_size= args.sampling_batch_size)

        curr = get_EDM(cov_dim=X[2].shape[1], hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                       data_len=data_len, num_epochs=args.n_epoch, lr=args.lr, batch_size=args.batch_size)

        curr.fit(ds_curr, epochs=args.n_epoch, verbose=1)
        prev = curr
        ds_prev = ds_curr

    X_test_intervened = X_test_original.copy()
    X_test_intervened[:, -1] = intervention[0]
    num_samples = 2048
    X_test_normalized = (X_test_intervened - ds_curr.x1_mean) / ds_curr.x1_std

    samples = curr.sample(X_test_normalized, num_samples).cpu().numpy()
    samples_normalized = samples * y_std + y_mean  # Rescale the samples
    plot_two_histograms_on_same(time_series_conditional_interventional_data(X_test_original[0], intervention), samples_normalized)


if __name__ == "__main__":
    size = 2**18 # 2**17

    X, Y = time_series_data(size, plot=True)
    X_test = np.array([[-2., 3. , 2. , -3.0, 2.0, 0.0]]) #np.array([[0., -2. , 0. ,4.0, 2.0, 0.0]])
    intervention = np.array([-5.0, -3.0, 0.0]) #np.array([5.0, 5.0, -2.0])

    time_series_conditional_interventional_data(X_test[0], intervention, sample_size=4096, plot=True)
    time_series_conditional_data(X_test[0], sample_size=4096, plot=True)

    estimate_iteratively_edm(X, Y)