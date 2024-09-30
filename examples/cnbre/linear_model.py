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

plt.style.use("computermodern")
# from clax.network import Network
dim = 50
np.random.seed(2024)
theta = 10
error = 1.0
model_0 = np.random.rand(dim, theta)
model_diff = np.random.rand(dim)
model_1 = np.concatenate((model_0, model_diff[..., None]), axis=-1)
mu_0 = np.zeros(theta)
mu_1 = np.zeros(theta + 1)
mu_1[-1] = 0.0
M_0 = LinearModel(M=model_0, C=np.eye(dim) * error, mu=mu_0)
M_1 = LinearModel(M=model_1, C=np.eye(dim) * error, mu=mu_1)


# def data_gen(n_samples, seed=2024, contrastive=True):

#     np.random.seed(seed)
#     n_sample = n_samples
#     theta = 10
#     error = 1.0
#     model_0 = np.random.rand(dim, theta)
#     model_diff = np.random.rand(dim)
#     model_1 = np.concatenate((model_0, model_diff[..., None]), axis=-1)
#     mu_0 = np.zeros(theta)
#     mu_1 = np.zeros(theta + 1)
#     mu_1[-1] = 1.0
#     M_0 = LinearModel(M=model_0, C=np.eye(dim) * error, mu=mu_0)
#     M_1 = LinearModel(M=model_1, C=np.eye(dim) * error, mu=mu_1)

#     t_0, d_0 = np.split(M_0.joint().rvs(n_sample // 2), [theta], axis=-1)
#     t_diff = M_1.prior().rvs(n_sample // 2)[..., -1]
#     t_1 = np.concatenate((t_0, t_diff[..., None]), axis=-1)
#     d_1 = M_1.likelihood(t_1).rvs()
#     X = np.moveaxis(np.stack([d_0, d_1]), 0, 2)
#     y = np.zeros(n_sample // 2)

#     x_test = np.concatenate([M_0.evidence().rvs(1000), M_1.evidence().rvs(1000)])

#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
#     return X, y, x_test, M_0, M_1

import jax.numpy as jnp


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


# optionally specify the optimizer manually
import optax

chain = optax.chain(
    optax.adaptive_grad_clip(0.01),
    optax.adamw(5e-3),
)


class Network(nn.Module):
    """A simple MLP classifier."""

    n_initial: int = 1028
    n_hidden: int = 128
    n_layers: int = 3
    n_out: int = 1
    # act = nn.silu

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.n_initial)(x)
        # hacky way to make batchnorm have no impact
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.silu(x)
        x = nn.Dense(self.n_out)(x)
        return x


dims = [int(1e3), int(1e4), int(1e5), int(1e6)]
true_ks = []
network_ks = []
network_ks_contrastive = []
lr = 1e-4
# X, y, X_test, M_0, M_1 = data_gen(dims[0], contrastive=True)
X_test = np.concatenate([M_0.evidence().rvs(5000), M_1.evidence().rvs(5000)])
# X_test = M_1.evidence().rvs(5000)

for i in dims:
    t_0, d_0 = np.split(M_0.joint().rvs(i // 2), [theta], axis=-1)
    t_diff = M_1.prior().rvs(i // 2)[..., -1]
    t_1 = np.concatenate((t_0, t_diff[..., None]), axis=-1)
    d_1 = M_1.likelihood(t_1).rvs()
    X = np.moveaxis(np.stack([d_0, d_1]), 0, 2)
    y = np.zeros(i // 2)

    classifier_contrastive = SDClassifier()
    classifier_contrastive.network = Network()
    classifier_contrastive.fit(
        X, y, epochs=1000, lr=lr, ndims=dim, batch_size=i // 20, optimizer=chain
    )

    d_0 = M_0.evidence().rvs(X.shape[0])
    d_1 = M_1.evidence().rvs(X.shape[0])
    X = np.concatenate((d_0, d_1))
    y = np.concatenate((np.zeros(d_0.shape[0]), np.ones(d_1.shape[0])))

    classifier = ClassifierSamples()
    classifier.network = Network()
    classifier.fit(
        d_1, d_0, epochs=1000, lr=lr, ndims=dim, batch_size=i // 10, optimizer=chain
    )
    true_k = M_1.evidence().logpdf(X_test) - M_0.evidence().logpdf(X_test)
    true_ks.append(true_k)

    network_k = classifier.predict(X_test).squeeze()
    network_k_contrastive = classifier_contrastive.predict(X_test).squeeze()
    print(f"RMSE: {np.sqrt(np.mean((true_k - network_k) ** 2))}")
    network_ks.append(network_k)
    network_ks_contrastive.append(network_k_contrastive)
    print(
        f"RMSE contrastive: {np.sqrt(np.mean((true_k - network_k_contrastive) ** 2))}"
    )
    f, a = plt.subplots()
    a.scatter(true_k, network_k, label="Vanilla", s=4)
    a.scatter(true_k, network_k_contrastive, label="CNBRE", s=4)
    a.plot((-1, 15), (-1, 15), color="black", linestyle="--")
    a.set_xlabel(r"True $\ln K$")
    a.set_ylabel(r"Network $\ln K$")
    a.legend()
    f.savefig(f"en_{i}.pdf")
    f, a = plt.subplots()
    a.plot(classifier.trace.losses, label="Vanilla")
    a.plot(classifier_contrastive.trace.losses, label="CNBRE")
    a.legend()
    a.set_yscale("log")
    a.set_xlabel("Epoch")
    a.set_ylabel("Loss")
    f.savefig(f"loss_{i}.pdf")


f, a = plt.subplots()

a.plot(
    dims,
    [
        np.sqrt(np.mean((true_k - network_k) ** 2))
        for true_k, network_k in zip(true_ks, network_ks)
    ],
    label="Vanilla",
    marker="o",
    markersize=4,
)
a.plot(
    dims,
    [
        np.sqrt(np.mean((true_k - network_k) ** 2))
        for true_k, network_k in zip(true_ks, network_ks_contrastive)
    ],
    label="CNBRE",
    marker="o",
    markersize=4,
)
a.set_xlabel("Number of samples")
a.set_ylabel("RMSE")
a.legend()
a.set_xscale("log")
a.set_yscale("log")
f.savefig("en.pdf")

# classifier.fit(X_train, y_train, epochs=200, lr=lr, ndims=dim, batch_size=1000)

# # X_test = jnp.concatenate([X_test[..., 0], X_test[..., 1]])
# true_k = M_1.evidence().logpdf(X_test) - M_0.evidence().logpdf(X_test)
# network_k = classifier.predict(X_test).squeeze()

# plt.scatter(true_k, network_k)
# print(f"RMSE: {np.sqrt(np.mean((true_k - network_k) ** 2))}")
# plt.plot((-10, 10), (-10, 10), color="black", linestyle="--")
# plt.xlabel(r"True $\ln K$")
# plt.ylabel(r"Network $\ln K$")
# plt.savefig("en.pdf")
