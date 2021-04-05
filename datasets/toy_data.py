import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import torch
import torch.distributions as tdist
from matplotlib import pyplot as plt


# For now just pinwheel is supported

def inf_train_gen(data, rng=None, n_points=200):
    if rng is None:
        rng = np.random.RandomState()
    
    if data == 'pinwheel':
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_points // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
