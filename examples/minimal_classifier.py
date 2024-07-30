"""
A minimal classifier example
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from optax import adam, apply_updates, sigmoid_binary_cross_entropy
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

np.random.seed(2024)
dim = 10
n_sample = 10000


m1 = np.random.randn(dim)
m2 = np.random.randn(dim)

M_0 = multivariate_normal(mean=m1, cov=np.eye(dim))
M_1 = multivariate_normal(mean=m2, cov=np.eye(dim))

X = np.concatenate((M_0.rvs(n_sample), M_1.rvs(n_sample)))
y = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


###############################################################################
# Neural Network code
# imports needed
# import jax
# from flax import linen as nn
# from optax import adam, apply_updates, sigmoid_binary_cross_entropy

rng = jax.random.PRNGKey(0)


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        for _ in range(2):
            x = nn.Dense(100)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


learning_rate = 1e-2
# epochs = steps * batch_size / n_sample
batch_size = 512
steps = 1000

network_params = Network().init(rng, X_train[0])
optimizer = adam(learning_rate=learning_rate)
state = optimizer.init(network_params)


def loss_fn(params, batch, labels):
    logits = Network().apply(params, batch)
    return sigmoid_binary_cross_entropy(logits.squeeze(), labels).mean()


for i in range(steps):
    rng, step_rng = jax.random.split(rng)
    idx = jax.random.choice(step_rng, X_train.shape[0], shape=(batch_size,))
    loss, grad = jax.value_and_grad(jax.jit(loss_fn))(
        network_params, X_train[idx], y_train[idx]
    )
    updates, state = optimizer.update(grad, state)
    network_params = apply_updates(network_params, updates)
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss}")


def predict(x):
    return Network().apply(network_params, x)


#########################################################################
# Check the outputs

network_log_k = predict(X_test).squeeze()
true_k = M_1.logpdf(X_test) - M_0.logpdf(X_test)


plt.scatter(network_log_k, true_k)
plt.show()
