"""
An example to compute a classifier two sample test

"""

import numpy as np
from jax.nn import sigmoid
from scipy.stats import multivariate_normal

from clax import ClassifierSamples

np.random.seed(2025)
dim = 100
n_sample = 10000


# m = np.zeros(dim)
m1 = np.random.randn(dim) * 0.1
m2 = np.random.randn(dim) * 0.1

M_0 = multivariate_normal(mean=m1, cov=np.eye(dim))
M_1 = multivariate_normal(mean=m2, cov=np.eye(dim))


D_0 = M_0.rvs(n_sample)
D_1 = M_1.rvs(n_sample)


classifier = ClassifierSamples()

classifier.fit(D_0, D_1, epochs=50, lr=1e-2, batch_size=128)

D_test = M_0.rvs(n_sample)

print(f"C2ST score: {np.mean(sigmoid(classifier.predict(D_test))):.2f}")
