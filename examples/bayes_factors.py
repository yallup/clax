"""
An example to compute Bayes factors

Demonstrate the importance of going slow when training a classifier

"""

import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from clax import Classifier

# from clax.network import Network

np.random.seed(2024)
dim = 100
n_sample = 500000


c1 = np.random.rand(dim) - 0.5
c2 = np.random.rand(dim) - 0.5


m1 = np.ones(dim) * 0.1
m2 = np.zeros(dim)
midpoint = (m1 + m2) / 2
error = 0.025

M_0 = multivariate_normal(mean=m1, cov=np.eye(dim) * error)
M_1 = multivariate_normal(mean=m2, cov=np.eye(dim) * error)
M_2 = multivariate_normal(mean=midpoint, cov=np.eye(dim) * error)

X = np.concatenate((M_0.rvs(n_sample), M_1.rvs(n_sample)))
y = np.concatenate((np.zeros(n_sample), np.ones(n_sample)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

classifier = Classifier()

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


lr = 1e-4
classifier.network = Network(n_out=1, n_initial=1056, n_hidden=128, n_layers=3)
classifier.fit(X_train, y_train, epochs=100, lr=lr, batch_size=10000)

true_k = M_1.logpdf(X_test) - M_0.logpdf(X_test)
network_k = classifier.predict(X_test).squeeze()

X_test_2 = M_2.rvs(n_sample // 50)
# X_test_2 = X_transform.transform(X_test_2)
true_k_m2 = M_1.logpdf(X_test_2) - M_0.logpdf(X_test_2)
network_k_m2 = classifier.predict(X_test_2).squeeze()


def plot():
    f, a = plt.subplots(1, 1, figsize=(6, 4))
    a.scatter(
        true_k_m2,
        network_k_m2,
        alpha=0.5,
        c="C1",
        label=r"$M_2$ test",
        marker=".",
        rasterized=True,
    )
    a.scatter(
        true_k,
        network_k,
        alpha=0.5,
        c="C0",
        label=r"$M_0 \cup M_1$ test",
        marker=".",
        rasterized=True,
    )
    # minmin=min(true_k.min(), network_k.min())
    # minmin=-50
    # maxmax=50
    maxrange = 50
    xrange = (-maxrange, maxrange * 1.5)
    yrange = (-maxrange * 1.5, maxrange)

    # maxmax=max(true_k.max(), network_k.max())
    a.plot(
        (-maxrange, maxrange),
        (-maxrange, maxrange),
        color="black",
        linestyle="--",
    )

    target = 10

    mask = np.logical_and(
        np.logical_and(true_k > -target, true_k < target),
        np.logical_and(network_k > -target, network_k < target),
    )
    mask_m2 = np.logical_and(
        np.logical_and(true_k_m2 > -target, true_k_m2 < target),
        np.logical_and(network_k_m2 > -target, network_k_m2 < target),
    )
    axins = zoomed_inset_axes(
        a, int(maxrange / target * 0.5 + 1), loc=4, borderpad=1.5
    )  # zoom = 6
    axins.scatter(
        true_k_m2[mask_m2],
        network_k_m2[mask_m2],
        c="C1",
        alpha=0.5,
        marker=".",
        rasterized=True,
    )
    axins.scatter(
        true_k[mask],
        network_k[mask],
        alpha=0.5,
        c="C0",
        marker=".",
        rasterized=True,
    )

    axins.plot(
        (-target, target),
        (-target, target),
        color="black",
        linestyle="--",
    )

    criticalrms = np.sqrt(np.mean((true_k_m2[mask_m2] - network_k_m2[mask_m2]) ** 2))

    # sub region of the original image
    axins.set_xlim(-target, target)
    axins.set_ylim(-target, target)
    axins.set_title(f"RMS: {criticalrms:.2f}", fontsize=6)

    # a.set_xlim((true_k.min(), true_k.max()))
    # a.set_ylim((network_k.min(), network_k.max()))

    a.set_ylim(yrange)
    a.set_xlim(xrange)

    # change the font size of the axins ticks
    plt.setp(axins.get_xticklabels(), fontsize=6)
    plt.setp(axins.get_yticklabels(), fontsize=6)
    a.legend(fontsize=6)

    # plt.xticks(visible=False)
    # plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(
        a,
        axins,
        loc1=1,
        loc2=3,
        fc="none",
        ec="0.5",
        linestyle="dotted",
        linewidth=0.5,
    )
    a.set_xlabel(r"True $\ln K$")
    a.set_ylabel(r"Network $\ln K$")
    f.tight_layout()
    f.savefig("en.pdf")


plot()

f, a = plt.subplots()
a.plot(classifier.trace.losses)
a.set_yscale("log")
f.savefig("losses.pdf")
