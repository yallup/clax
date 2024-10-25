# A minimal classifier example with time interpolation (t) and proper batchhandling

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from optax import adam, apply_updates, sigmoid_binary_cross_entropy, adamw
import optax
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

np.random.seed(2024)
dim = 10
n_sample = 100000

# Generate means for two classes
m1 = np.random.randn(dim)
m2 = np.random.randn(dim)

# Multivariate normal distributions
M_0 = multivariate_normal(mean=m1, cov=np.eye(dim))
M_1 = multivariate_normal(mean=m2, cov=np.eye(dim))

# Sample data
X_class0 = M_0.rvs(n_sample)
X_class1 = M_1.rvs(n_sample)
y_class0 = np.zeros(n_sample)
y_class1 = np.ones(n_sample)

X = np.concatenate((X_class0, X_class1), axis=0)
y = np.concatenate((y_class0, y_class1), axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, stratify=y
)

# Convert to JAX arrays
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

###############################################################################
# Neural Network code

rng = jax.random.PRNGKey(0)


class Network(nn.Module):
    @nn.compact
    def __call__(self, x, t):
        # Concatenate the time parameter t to the input x
        t = t.reshape(-1, 1)  # Ensure t has shape (batch_size, 1)
        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(50)(x)
        x = nn.silu(x)
        for _ in range(2):
            x = nn.Dense(10)(x)
            x = nn.silu(x)
        x = nn.Dense(dim)(x)
        return x.squeeze()  # Return logits


learning_rate = 1e-3
# batch_size = 256
batch_size = 512
steps = 1000

# Initialize network parameters and optimizer state
network_init = Network().init(rng, X_train[:1], jnp.array([0.0]))
network_params = network_init
optimizer = adamw(learning_rate=learning_rate)
state = optimizer.init(network_params)

# network_init_baseline = Network().init(rng, X_train[:1])  # , jnp.array([0.0]))
# network_params_baseline = network_init_baseline
# optimizer_baseline = adamw(learning_rate=learning_rate)
# state_baseline = optimizer_baseline.init(network_params_baseline)


def loss_fn(params, batch, rng, alpha=1.0):
    # batch has shape (batch_size, 2, dim)
    x0 = batch[:, 0, :]  # Samples from class 0
    x1 = batch[:, 1, :]  # Samples from class 1

    # Generate random t between 0 and 1
    rng, t_rng = jax.random.split(rng)
    t = jax.random.uniform(t_rng, shape=(x0.shape[0],1))
    # t = jnp.power(t, alpha)
    rng, n_rng = jax.random.split(rng)
    # N_batch = x0.shape[0]
    # t = jax.random.uniform(rng, (N_batch, 1))
    noise = jax.random.normal(rng, x0.shape)
    psi_0 = t * x1 + (1 - t) * x0 + 1e-3 * noise
    output = Network().apply(params, psi_0, t)
    psi = x1 - x0
    return jnp.mean((output - psi) ** 2), rng

    # rng, c_rng = jax.random.split(rng)
    # c = jax.random.choice(c_rng, 2, shape=(x0.shape[0],), replace=True)

    # # Linearly interpolate between samples
    # x_t = t[:, None] * x1 + (1 - t)[:, None] * x0
    # # Interpolated labels (since class labels are 0 and 1)
    # y_t = t  # Because y0 = 0 and y1 = 1

    # x_t_inv = (1 - t)[:, None] * x1 + t[:, None] * x0
    # y_t_inv = 1 - t
    # x_t = jnp.concatenate([x_t, x_t_inv], axis=0)
    # y_t = jnp.concatenate([y_t, y_t_inv], axis=0)

    # # Forward pass through the network
    # logits = Network().apply(params, x_t)
    # # Compute binary cross-entropy loss
    # loss = sigmoid_binary_cross_entropy(logits, y_t).mean()
    # return loss, rng  # Return updated rng


losses_baseline = []
losses = []
# Training loop
for step in range(steps):
    rng, data_rng = jax.random.split(rng)
    # Sample indices for class 0 and class 1
    idx0 = jax.random.choice(
        data_rng,
        X_train[y_train == 0].shape[0],
        shape=(batch_size,),
        replace=False,
    )
    idx1 = jax.random.choice(
        data_rng,
        X_train[y_train == 1].shape[0],
        shape=(batch_size,),
        replace=False,
    )
    x0_batch = X_train[y_train == 0][idx0]
    x1_batch = X_train[y_train == 1][idx0]

    # Stack batches to form batch of shape (batch_size, 2, dim)
    batch = jnp.stack([x0_batch, x1_batch], axis=1)

    # Compute loss and gradients (handle rng inside loss function)
    # with jax.disable_jit():
    (loss, rng), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            network_params, batch, rng, alpha=2
        )

    # Update parameters
    updates, state = optimizer.update(grad, state, network_params)
    network_params = apply_updates(network_params, updates)

    # (loss_baseline, rng), grad = jax.value_and_grad(loss_fn, has_aux=True)(
    #     network_params_baseline, batch, rng, alpha=1000.0
    # )
    # updates_baseline, state_baseline = optimizer_baseline.update(grad, state_baseline, network_params_baseline)
    # network_params_baseline = apply_updates(network_params_baseline, updates_baseline)

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss}")
    losses.append(loss)
    # losses_baseline.append(loss_baseline)


def predict(x):
    # At inference time, set t = 1
    t = jnp.ones((x.shape[0],))
    logits = Network().apply(network_params, x)
    # probs = jax.nn.sigmoid(logits)
    # return probs
    return logits


def predict_baseline(x):
    # At inference time, set t = 1
    t = jnp.ones((x.shape[0],))
    logits = Network().apply(network_params_baseline, x)
    # probs = jax.nn.sigmoid(logits)
    # return probs
    return logits


def likelihood(t):
    logits = Network().apply(network_params, X_test, t)
    return jnp.mean((logits - X_test)**2)
from jaxopt import LBFGS
t_init = jax.random.normal(rng, (X_train.shape[0], 1))

tuner = LBFGS(fn)
res = tuner.run(t_init)
t_init
###############################################################################
# Check the outputs

network_log_k = predict(X_test).squeeze()
true_k = M_1.logpdf(X_test) - M_0.logpdf(X_test)

network_baseline_log_k = predict_baseline(X_test).squeeze()

# Create a figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Panel 1: Scatter plot of network output vs. true log odds
ax1.scatter(network_log_k, true_k, alpha=0.5, label=r"$\alpha=2$")
ax1.scatter(network_baseline_log_k, true_k, alpha=0.5, label=r"$\alpha=1000$")
ax1.set_xlabel("Network Log-Odds")
ax1.legend()
ax1.set_ylabel("True Log-Odds")
ax1.set_title("Comparison of Network Output and True Log-Odds")
ax1.plot([-10, 10], [-10, 10], "r--")  # Line y=x for reference

# Panel 2: Loss curve during training (dummy example)
# loss_values = [0.5, 0.4, 0.3, 0.2, 0.1]
ax2.plot(losses, label="Training Loss")
ax2.plot(losses_baseline, label="Training Loss Baseline")
ax2.set_xlabel("Epoch")
# ax2.legend()
ax2.set_ylabel("Loss")
ax2.set_yscale("log")
ax2.set_title("Training Loss Curve")
fig.savefig("fm_output.pdf")
plt.show()
