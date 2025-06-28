import numpy as np
import matplotlib.pyplot as plt

def plot_two_histograms_on_same(true_distr, estimated_distr):
    if true_distr.ndim == 1:
        plt.figure(figsize=(10, 6))
        plt.hist(true_distr, bins=100, density=True, alpha=0.5, label='True Distribution', color='blue')
        plt.hist(estimated_distr, bins=100, density=True, alpha=0.5, label='Estimated Distribution', color='orange')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()
    elif true_distr.ndim == 2:
        plt.figure(figsize=(10, 6))
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.hist(true_distr[:,i], bins=100, density=True, alpha=0.5, label='True Distribution', color='blue')
            plt.hist(estimated_distr[:,i], bins=100, density=True, alpha=0.5, label='Estimated Distribution', color='orange')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid()
        plt.show()
    else:
        raise ValueError("Distributions must be 1D or 2D arrays for histogram plotting.")

def plot_histogram(samples):
    if samples.ndim == 1:
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=100, density=True, alpha=0.6, color='g')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid()
        plt.show()
    elif samples.ndim == 2:
        #plot the dimensions separately
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(samples[:,0], bins=100, density=True, alpha=0.6, color='g')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.hist(samples[:,1], bins=100, density=True, alpha=0.6, color='g')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid()
        plt.show()
    else :
        raise ValueError("Samples must be 1D or 2D array for histogram plotting.")