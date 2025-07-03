import os
# specify the keras backend we want to use
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops
import torch
from sklearn.datasets import make_moons

import numpy as np
import scipy
import matplotlib.pyplot as plt

#keras.backend.set_floatx('float32')

def c_skip_fn(sigma, sigma_data):
    return sigma_data**2 / (sigma**2 + sigma_data**2)

def c_out_fn(sigma, sigma_data):
    return sigma * sigma_data / ops.sqrt(sigma_data**2 + sigma**2)

def c_in_fn(sigma, sigma_data):
    return 1 / ops.sqrt(sigma**2 + sigma_data**2)

def c_noise_fn(sigma):
    return 0.25 * ops.log(sigma)
def compute_loss(network, x, y, sigma_data, seed_generator, training=False):
    # hyper-parameters for sampling the time
    p_mean = -1.2
    p_std = 1.2

    # sample log-noise
    log_sigma = p_mean + p_std * keras.random.normal(
        ops.shape(x)[:1], dtype=ops.dtype(x), seed=seed_generator
    )
    # noise level with shape (batch_size, 1)
    sigma = ops.exp(log_sigma)[:, None]

    # generate noise vector
    z = sigma * (keras.random.normal(
        ops.shape(x), dtype=ops.dtype(x), seed=seed_generator
    ))

    # calculate preconditioning
    c_skip = c_skip_fn(sigma, sigma_data)
    c_out = c_out_fn(sigma, sigma_data)
    c_in = c_in_fn(sigma, sigma_data)
    c_noise = c_noise_fn(sigma)
    lam = 1 / c_out[:,0]**2

    # calculate output of the network
    inp = ops.concatenate([c_in * (x + z), y, c_noise], axis=-1)
    out = network(inp, training=training)

    #  calculate loss

    # for the given c_out and lambda, this is just one
    effective_weight = lam * c_out[:,0]**2
    unweighted_loss = ops.mean((out - 1/c_out * (x - c_skip * (x + z)))**2,axis=-1)
    loss = ops.mean(effective_weight * unweighted_loss)

    # so we could also write
    ## loss = ops.mean((out - 1/c_out * (x - c_skip * (x + z)))**2)

    return loss

class EDM(keras.Model):
    def __init__(self, output_dim, cov_dim, hidden_dim=256, n_hidden=4, sigma_data=1.0, prev_model=None, name="trigflow", **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        input_shape = (output_dim + cov_dim + 1,)
        self.network = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                # n_hidden hidden layers
                *(keras.layers.Dense(hidden_dim, activation="mish") for i in range(n_hidden)),
                keras.layers.Dense(output_dim, bias_initializer="zeros", kernel_initializer="zeros"),
            ]
        )

        self.sigma_data = sigma_data
        self.seed_generator = keras.random.SeedGenerator(seed=2024)
        self.prev_model = prev_model  # For iterative training

    def build(self, input_shape):
        self.network.build(input_shape)

    def call(self, inputs):
        return self.network(inputs)


    # write a loss for PyTorch, see the following link for instructions for other backends
    # https://keras.io/guides/custom_train_step_in_torch/#a-first-simple-example
    def train_step(self, data):
        covariate, cov2_or_output = data

        # If prev_model is set, use it for iterative training
        #if self.prev_model is not None:
        #    with torch.no_grad():
        #        cov2_or_output = keras.ops.stop_gradient(self.prev_model.sample(cov2_or_output, sample_each=1))
        
        loss = compute_loss(
            self.network, cov2_or_output, covariate, self.sigma_data, self.seed_generator, training=True
        )
        self.zero_grad()
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    

    def sample(self, y_cond, sample_each=1, atol=1e-5, rtol=1e-5, z=None):
        max_t = 80.0
        min_t = 1e-4

        y_cond = ops.convert_to_tensor(y_cond, dtype="float32")
        y_cond = y_cond.repeat(sample_each, 1)

        #print(f"y_cond shape: {y_cond.shape}")
        batch_size = y_cond.shape[0]
        
        # Sample from the latent distribution
        if z is None:
            # generate noise vector
            z = max_t * (keras.random.normal(
                (batch_size, self.output_dim), seed=self.seed_generator
            ))

        shape = z.shape
        # set start time
        t = ops.ones(shape[:1]) * max_t

        def velocity_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = ops.reshape(ops.convert_to_tensor(sample, dtype="float32"), shape)
            time_steps = ops.reshape(ops.convert_to_tensor(time_steps,dtype="float32"), (-1, 1))
            with torch.no_grad():
                v = velocity(self, sample, y_cond, time_steps)
            return ops.reshape(v, (-1,)).cpu().numpy()

        def ode_func(t, x):
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t
            return velocity_wrapper(x, time_steps)

        # Use scipy blackbox ODE solver
        res = scipy.integrate.solve_ivp(
            ode_func,
            (max_t, min_t),
            z.cpu().numpy().reshape(-1),
            rtol=rtol,
            atol=atol,
            method='RK45'
        )
        #print(f"Number of function evaluations: {res.nfev}")
        x = ops.reshape(ops.convert_to_tensor(res.y[:, -1], dtype="float32"), shape)
        return x
    
def denoiser_fn(model, x, y, sigma):
    inp = ops.concatenate([c_in_fn(sigma, model.sigma_data) * x, y, c_noise_fn(sigma)], axis=-1)
    out = model(inp, training=False)
    return c_skip_fn(sigma, model.sigma_data) * x + c_out_fn(sigma, model.sigma_data) * out

def velocity(model, x, y, t):
    """Compute the velocity for the PF-ODE"""
    d = denoiser_fn(model, x, y, t)
    return (x - d) / t

def sample_target(n_samples):
    return make_moons(n_samples, noise=0.06)

def get_EDM(cov_dim, hidden_dim, n_hidden, data_len, num_epochs, lr, batch_size):

    n_total = num_epochs * data_len // batch_size  # Total number of gradient updates
    scheduled_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=n_total,
        alpha=1e-8
    )
    optimizer = keras.optimizers.Adam(learning_rate=scheduled_lr)
    prev = EDM(output_dim=1, cov_dim=cov_dim, n_hidden=n_hidden, hidden_dim=hidden_dim)  # sigma_data=ops.convert_to_tensor(ds.y_std.astype(np.float32))
    prev.compile(optimizer)
    prev.build((None, cov_dim + 1 + 1))
    return prev

def main():
    n_samples = 2000
    target_samples, labels = sample_target(n_samples)
    plt.scatter(target_samples[:,0], target_samples[:,1], s=2, c=labels, cmap='viridis')
    plt.show()
    batch_size = 256
    n_batches = 128

    train_data, labels = sample_target(batch_size * n_batches)
    labels = labels[:, None]  
    #convert train_data to 32 bit float tensor

    train_dataset = torch.utils.data.TensorDataset(
        ops.convert_to_tensor(labels.astype(np.float32)),  ops.convert_to_tensor(train_data.astype(np.float32))
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    #get first batch
    train_data_batch, labels_batch = next(iter(train_dataloader))
    print(f"train_data_batch shape: {train_data_batch.shape}, labels_batch shape: {labels_batch.shape}")
    print(f'first 10 labels : {labels_batch[:10,:].cpu().numpy()}')
    n_epochs = 10
    initial_learning_rate = 1e-3

    # total number of gradient updates
    n_total = n_epochs * n_batches
    # decay learning rate over time
    scheduled_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=n_total,
        alpha=1e-8
    )
    optimizer = keras.optimizers.Adam(learning_rate=scheduled_lr)
    edm = EDM(output_dim=2, cov_dim=1, n_hidden=6, hidden_dim=256)
    edm.compile(optimizer)
    edm.build((None, 4))
    
    history = edm.fit(train_dataloader, epochs=n_epochs, verbose=2)
    plt.plot(history.history["loss"], marker="x")
    plt.xlabel("epoch")
    _ = plt.ylabel("loss")
    plt.show()
    
    # Final sample generation
    sample_labels_one = np.ones((2000, 1)) 
    sample_labels_zero = np.zeros((2000, 1)) 

    samples = edm.sample(sample_labels_zero).cpu().numpy()

    _ = plt.scatter(samples[:, 0], samples[:, 1], s=2)
    plt.show()

if __name__ == "__main__":
    main()
    