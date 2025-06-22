# Creating a minimal dataset where there is history dependency.

import numpy as np
from treeffuser import Treeffuser
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



def conditional_data(X, sample_size=1000):
    A_0, A_1, A_2 = X[3], X[4], X[5]
    X_0, X_1, X_2 = X[0], X[1], X[2]

    m = 30
    means = np.array([-m + A_0, A_1, m + A_2])  # shape (3,)

    components = np.random.choice(3, size=sample_size)

    Y = np.zeros( (sample_size, 2))
    Y[:, 0] = np.array([np.random.normal(loc=means[comp], scale=1.0) for comp in components])
    Y[:, 1] = np.random.normal(loc=X_0 + X_1 + X_2, scale=1.0, size=sample_size)
    plot_histogram(Y)
    return Y

def iterative_conditional_data(X, intervention, sample_size=1000):
    A_0, A_1, A_2 = X[3], X[4], intervention[0]
    X_0, X_1, X_2 = X[0], X[1], X[2]

    A_3, A_4 = intervention[1], intervention[2]
    X_3 = X_0 + A_0 + X_1 * A_1 + X_2 * A_2  # this is the implementation of x_3 = x_0 + a_0 + x_1 * a_1 + x_2 * a_2 in backward direction
    X_4 = X_1 + A_1 + X_2 * A_2 + X_3 * A_3

    # do two iterations 
    A_0, A_1, A_2 = A_2, A_3, A_4  
    X_0, X_1, X_2 = X_2, X_3, X_4

    print(A_0, A_1, A_2)

    m = 30
    means = np.array([-m + A_0, A_1, m + A_2])  # shape (3,)

    components = np.random.choice(3, size=sample_size)

    Y = np.zeros( (sample_size, 2))
    Y[:, 0] = np.array([np.random.normal(loc=means[comp], scale=1.0) for comp in components])
    Y[:, 1] = np.random.normal(loc=X_0 + X_1 + X_2, scale=1.0, size=sample_size)
    #plot_histogram(Y)
    return Y

def base_data(size, plot=False):
    # Create the dataset with gaussian noise
    A_0, A_1, A_2 = (np.random.normal(loc=0.0, scale=5, size=size) for _ in range(3))
    X_0, X_1, X_2 = (np.random.normal(loc=0.0, scale=3.0, size=size) for _ in range(3))

    m = 30
    means = np.stack([-m + A_0, A_1, m + A_2], axis=1)

    components = np.random.choice(3, size=size)

    Y = np.zeros( (size, 2))
    Y[:, 0] = np.array([
    np.random.normal(loc=means[i, comp], scale=1.0) for i, comp in enumerate(components) ])

    Y[:, 1] = np.random.normal(loc=X_0 + X_1 + X_2, scale=1.0)

    X = np.column_stack((X_0, X_1, X_2, A_0, A_1, A_2))
    if plot:
        plot_histogram(Y)

    return X, Y

def iterative_data(X):
    res = np.zeros((3, *X.shape))
    X_val = X[:, :3]
    A_val = X[:, 3:6]

    res[0] = X
    for i in range(2):
        new_A_values = A_val[: , [2, 0, 1]] # this is the implementation of a_3 = a_0
        new_X_values = np.zeros_like(X_val)
        new_X_values[:,0] = X_val[:, 2] - A_val[:, 2] - X_val[:, 0] * A_val[:, 0] - X_val[:, 1] * A_val[:, 1] # this is the implementation of x_3 = x_0 + a_0 + x_1 * a_1 + x_2 * a_2 in backward direction
        new_X_values[:, [1,2]] = X_val[:, [0,1]]
        res[i+1] = np.column_stack((new_X_values, new_A_values))
    
    return res

def estimate_basic(X, Y):
    seed = 0
    treeffuser = Treeffuser(seed=seed)
    print(f"shape of X: {X.shape}, shape of Y: {Y.shape}")
    treeffuser.fit(X, Y)

    X_test = np.array([[-3., -3. ,-3. , 5.0, -5.0, 0.0]])
    num_samples = 10000
    samples = treeffuser.sample(X_test, num_samples).reshape(num_samples, 2)
    plot_two_histograms_on_same(conditional_data(X_test[0]), samples)

def estimate_iteratively(X, Y):
    seed = 0
    X_test = np.array([[-2., 3. , 2. , -3.0, 2.0, 0.0]]) #np.array([[0., -2. , 0. ,4.0, 2.0, 0.0]])
    intervention = np.array([1.0, -1.0, 2.0]) #np.array([5.0, 5.0, -2.0])

    iterative_conditional_data(X_test[0], intervention)
    return

    print("fitting the base model")
    prev = Treeffuser(seed=seed)
    prev.fit(X[0], Y)

    for i in range(2):
        print(f"Iteration {i+1}")
        intervened_data = X[i]
        intervened_data[:, -1] = intervention[2-i]
        Y = prev.sample(intervened_data, 1)
        Y = Y.reshape(-1, 2)
        curr = Treeffuser(seed=seed)
        curr.fit(X[i+1], Y)
        prev = curr

    print("Sampling from the final model")
    X_test_intervened = X_test.copy()
    X_test_intervened[:, -1] = intervention[0]
    num_samples = 10000
    samples = curr.sample(X_test_intervened, num_samples).reshape(num_samples, 2)
    plot_two_histograms_on_same(iterative_conditional_data(X_test[0], intervention), samples)

if __name__ == "__main__":
    size = 100000

    X, Y = base_data(size, plot=True)
    #estimate_basic(X, Y)
    
    X = iterative_data(X)
    estimate_iteratively(X, Y)