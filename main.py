# Creating a minimal dataset where there is history dependency.

import os
# specify the keras backend we want to use
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
#from treeffuser import Treeffuser
import matplotlib.pyplot as plt
from diffusion_model import DDPM
import argparse
from edm import EDM
from dataset import HistoryDataset
import keras
from keras import ops

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

def create_data(size, mixture = True, plot=False):
    # Create the dataset with gaussian noise
    A_0, A_1, A_2 = (np.random.normal(loc=0.0, scale=5, size=size) for _ in range(3))#(np.random.uniform(low = -10, high = 10, size=size) for _ in range (3)) #
    m = 30

    if mixture:
        means = np.stack([-m + A_0 , A_1 , m + A_2], axis=1)  #
        means = np.stack([-m + A_0 * A_1 + A_2 ** 2 + 2 * A_0 * A_1, A_1, m + A_2 + 5*A_0 + A_0 * A_1 * A_2], axis=1)
        components = np.random.choice(3, size=size)
        Y = np.array([np.random.normal(loc=means[i, comp], scale=1.0) for i, comp in enumerate(components)])
    else :
        Y = np.random.normal(loc = m + A_0 + A_1 + A_2, scale=1.0, size=size)  

    X = np.column_stack((A_0, A_1, A_2))
    if plot:
        plot_histogram(Y)
    
    return X, Y

def conditional_data(X, sample_size=5000, mixture=True, plot=False):
    A_0, A_1, A_2 = X[0], X[1], X[2]
    m = 30
    if mixture:
        means = np.array([-m + A_0, A_1, m + A_2])  # shape (3,). 
        means = np.array([-m + A_0 * A_1 + A_2 ** 2 + 2 * A_0 * A_1, A_1, m + A_2 + 5*A_0 + A_0 * A_1 * A_2]) 
        components = np.random.choice(3, size=sample_size)
        Y = np.array([np.random.normal(loc=means[comp], scale=1.0) for comp in components])

    else:
        Y = np.random.normal(loc=m + A_0 +  A_1 + A_2, scale=1.0, size=sample_size)  # shape (sample_size,)

    if plot:
        plot_histogram(Y)

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
        'n_epoch': 50,
        'model_name': 'basic_nn',
        'hidden_dim': 256,
        'batch_size': 256,
        'n_T': 2000,
        'lr': 1e-3,
        'train': False  # Set to True to train the model
        }
    
    args = argparse.Namespace(**args_dict)
   
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    ddpm = DDPM(cov_dim=X.shape[1], hidden_dim=args.hidden_dim, betas=(1e-4, 0.02), device=device, n_T=args.n_T)
    if args.train:
        ddpm.to(device)
        ddpm.fit(args, X, Y)
        #load the model from the file
    else:
        ddpm.load_state_dict(torch.load('models/basic_nn.pth'))
        ddpm = ddpm.to(device)
    X_test = np.array([[5.0, -5.0, 0.0]])
    samples = ddpm.predict( X_test,  sample_size=2048)
    samples_denormalized = samples * ddpm.y_std.cpu().numpy() + ddpm.y_mean.cpu().numpy()  # Rescale the samples

    plot_two_histograms(conditional_data(X_test[0], 5000), samples_denormalized)

def estimate_basic_edm(X, Y):
    batch_size = 256
    Y = Y[:, None]
    ds = HistoryDataset(X, Y, keras=True)
    training_batched = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True )

    n_epochs = 50  # Reduced for faster testing
    initial_learning_rate = 1e-3
    n_total = n_epochs * Y.shape[0] // batch_size  # Total number of gradient updates
    scheduled_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=n_total,
        alpha=1e-8
    )
    optimizer = keras.optimizers.Adam(learning_rate=scheduled_lr)
    edm = EDM(output_dim=Y.shape[1], cov_dim=X.shape[1], n_hidden=6, hidden_dim=256)  # sigma_data=ops.convert_to_tensor(ds.y_std.astype(np.float32))
    edm.compile(optimizer)
    edm.build((None, X.shape[1] + Y.shape[1] + 1))

    print(f"Model device: {next(edm.parameters()).device}")
    print(f"Training on {len(training_batched)} batches per epoch")
    
    history = edm.fit(training_batched, epochs=n_epochs, verbose=1)
    X_test_original = np.array([[5.0, -5.0, 0.0]])
    X_test_normalized = (X_test_original - ds.x_mean) / ds.x_std
    
    print(f"Original test point: {X_test_original[0]}")
    print(f"Normalized test point: {X_test_normalized[0]}")
    print(f"Data mean: {ds.x_mean}")
    print(f"Data std: {ds.x_std}")

    samples = edm.sample(X_test_normalized, sample_each=2048).cpu().numpy()
    samples = samples * ds.y_std + ds.y_mean  # Rescale the samples

    print(f"Samples shape: {samples.shape}")
    
    # Use original unnormalized point for true distribution
    true_samples = conditional_data(X_test_original[0], 2048)
    print(f"True samples mean: {true_samples.mean():.3f}, std: {true_samples.std():.3f}")
    print(f"Generated samples mean: {samples.mean():.3f}, std: {samples.std():.3f}")
    
    plot_two_histograms(true_samples, samples)

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
    samples_denormalized = samples * prev.y_std + prev.y_mean  # Rescale the samples

    plot_two_histograms(conditional_data(X_test[0]), samples_denormalized)

if __name__ == "__main__":
    size = 100000  # Reduced from 1000000 for faster training
    X, Y = create_data(size)
    X_test = np.array([[5.0, -5.0, 0.0]])
    #conditional_data(X_test[0])
    #estimate_basic(X, Y)
    estimate_basic_edm(X, Y)