# Creating a minimal dataset where there is history dependency.

import torch
import numpy as np
from treeffuser import Treeffuser
import matplotlib.pyplot as plt
from diffusion_model import DDPM
import argparse

def plot_two_histograms(true_distr, estimated_distr):
    plt.figure(figsize=(10, 6))
    plt.hist(true_distr, bins=100, density=True, alpha=0.5, label='True Distribution', color='blue')
    plt.hist(estimated_distr, bins=100, density=True, alpha=0.5, label='Estimated Distribution', color='orange')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

def plot_histogram(samples):
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='g')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid()
    plt.show()

def create_data(size, plot=False):
    # Create the dataset with gaussian noise
    A_0, A_1, A_2 = (np.random.uniform(low = -10, high = 10, size=size) for _ in range (3)) #(np.random.normal(loc=0.0, scale=5, size=size) for _ in range(3))
    m = 30
    means = np.stack([-m + A_0 , A_1 , m + A_2], axis=1) #np.stack([-m + A_0 * A_1 + A_2 ** 2 + 2 * A_0 * A_1, A_1, m + A_2 + 5*A_0 + A_0 * A_1 * A_2], axis=1)

    components = np.random.choice(3, size=size)
    Y = np.array([
    np.random.normal(loc=means[i, comp], scale=1.0) for i, comp in enumerate(components) ])

    X = np.column_stack((A_0, A_1, A_2))
    if plot:
        plot_histogram(Y)
    
    return X, Y

def conditional_data(X, sample_size=1000):
    A_0, A_1, A_2 = X[0], X[1], X[2]
    m = 30
    means = np.array([-m + A_0, A_1, m + A_2]) #np.array([-m + A_0 * A_1 + A_2 ** 2 + 2 * A_0 * A_1, A_1, m + A_2 + 5*A_0 + A_0 * A_1 * A_2])  # shape (3,)

    components = np.random.choice(3, size=sample_size)
    Y = np.array([np.random.normal(loc=means[comp], scale=1.0) for comp in components])
    return Y

def iterative_data(X):
    new_X = np.zeros((3, *X.shape))
    new_X[0] = X[:, :]
    new_X[1] = X[:, [2,0,1]]
    new_X[2] = X[:, [1,2,0]]
    return new_X

def estimate_basic(X, Y):
    seed = 0
    treeffuser = Treeffuser(seed=seed, verbose=1)
    treeffuser.fit(X, Y)

    X_test = np.array([[5.0, -5.0, 0.0]])
    samples = treeffuser.sample(X_test, 10000)
    plot_two_histograms(conditional_data(X_test[0]), samples)

def estimate_basic_nn(X, Y):
    args_dict = {
        'n_epoch': 30,
        'model_name': 'basic_nn',
        'hidden_dim': 512,
        'batch_size': 256,
        'n_T': 200,
        'lr': 1e-5
        }
   
    args = argparse.Namespace(**args_dict)
   
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    ddpm = DDPM(cov_dim=X.shape[1], hidden_dim=args.hidden_dim, betas=(1e-4, 0.02), device=device, n_T=args.n_T, drop_prob=0.1)
    ddpm.to(device)
    ddpm.fit(args, X, Y)

    X_test = np.array([[5.0, -5.0, 0.0]])
    samples = ddpm.predict(X = X_test, sample_size=1000, guide_w=4.0)
    plot_two_histograms(conditional_data(X_test[0]), samples)

def estimate_iteratively(X, Y):
    seed = 0
    X_test = np.array([[5.0, -5.0, 0.0]])
    intervention = X_test # because of the simple data setting
    prev = Treeffuser(seed=seed)
    prev.fit(X[0], Y)

    for i in range(2):
        print(f"Iteration {i+1}")
        intervened_data = X[i]
        intervened_data[:, 2] = intervention[:, 2-i]
        Y = prev.sample(intervened_data, 1)
        Y = Y.flatten()
        curr = Treeffuser(seed=seed)
        curr.fit(X[i+1], Y)
        prev = curr

    print("Sampling from the final model")
    X_test_intervened = X_test[:, [1,2,0]]
    X_test_intervened[:, 2] = intervention[:,0]
    samples = curr.sample(X_test_intervened, 1000)

    plot_two_histograms(conditional_data(X_test[0]), samples)

if __name__ == "__main__":
    size = 1000000
    X, Y = create_data(size, plot=False)
    estimate_basic_nn(X, Y)