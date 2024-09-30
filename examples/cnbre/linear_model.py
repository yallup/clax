"""
An example to compute Bayes factors

Demonstrate the importance of going slow when training a classifier

"""

import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from lsbi.model import LinearModel
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from clax import Classifier, ClassifierSamples

# from clax.network import Network

np.random.seed(2024)
dim = 50
n_sample = 10000
theta = 10
error = 1.0
model_0 = np.random.rand(dim, theta)
model_diff = np.random.rand(dim)
model_1 = np.concatenate((model_0, model_diff[..., None]), axis=-1)

M_0 = LinearModel(M=model_0, C=np.eye(dim) * error)
M_1 = LinearModel(M=model_1, C=np.eye(dim) * error)

t_0, d_0 = np.split(M_0.joint().rvs(n_sample // 2), [theta], axis=-1)
t_diff = M_1.prior().rvs(n_sample // 2)[..., -1]
t_1 = np.concatenate((t_0, t_diff[..., None]), axis=-1)
d_1 = M_1.likelihood(t_1).rvs()

# d_0 = M_0.evidence().rvs(n_sample // 2)
# d_1 = M_1.evidence().rvs(n_sample // 2)

# X = np.concatenate((d_0, d_1))
# y = np.concatenate((np.zeros(n_sample // 2), np.ones(n_sample // 2)))

X = np.moveaxis(np.stack([d_0, d_1]), 0, 2)
y = np.zeros(n_sample // 2)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


class SDClassifier(Classifier):
    def loss(self, params, batch_stats, batch, labels, rng):
        """Loss function for training the classifier."""
        labels = jnp.concatenate((jnp.zeros_like(labels), jnp.ones_like(labels)))
        # batch = batch.reshape(-1, batch.shape[-1])
        batch = jnp.concatenate([batch[..., 0], batch[..., 1]])
        # batch = batch.reshape(batch.shape[0], -1)

        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            batch,
            train=True,
            mutable=["batch_stats"],
        )
        loss = self.loss_fn(output.squeeze(), labels).mean()
        return loss, updates


classifier = SDClassifier()
# classifier = Classifier()


# optionally specify the optimizer manually
# chain = optax.chain(
#     optax.adaptive_grad_clip(1.0),
#     optax.adamw(1e-3),
# )


class Network(nn.Module):
    """A simple MLP classifier."""

    n_initial: int = 256
    n_hidden: int = 64
    n_layers: int = 3
    n_out: int = 1
    # act = nn.silu

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.n_initial)(x)
        # hacky way to make batchnorm have no impact
        nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = nn.silu(x)
        x = nn.Dense(self.n_out)(x)
        return x


lr = 1e-3
classifier.network = Network(n_out=1, n_initial=256, n_hidden=64, n_layers=3)
import jax
import jax.numpy as jnp


def ExpLoss(logits, labels):
    return jnp.exp((labels.astype(jnp.float32) * -2 + 1.0) * logits).mean()


# classifier.loss_fn = ExpLoss

# with jax.disable_jit():
#     classifier.fit(X_train, y_train, epochs=100, lr=lr, ndims=dim, batch_size=512)

classifier.fit(X_train, y_train, epochs=100, lr=lr, ndims=dim, batch_size=512)

X_test = jnp.concatenate([X_test[..., 0], X_test[..., 1]])
true_k = M_1.evidence().logpdf(X_test) - M_0.evidence().logpdf(X_test)
network_k = classifier.predict(X_test).squeeze()

plt.scatter(true_k, network_k)
print(f"RMSE: {np.sqrt(np.mean((true_k - network_k) ** 2))}")
plt.plot((-10, 10), (-10, 10), color="black", linestyle="--")
plt.xlabel(r"True $\ln K$")
plt.ylabel(r"Network $\ln K$")
plt.savefig("en.pdf")
