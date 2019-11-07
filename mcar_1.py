import numpy as np
from matplotlib import pyplot as plt
import torch

np.random.RandomState(seed=0)

# reading in ribo data
data = np.genfromtxt('riboflavinV10.csv', delimiter=',', skip_header=1)
data = np.delete(data, 0, axis=1)  # gets rid of the row title (ribo_X)
data_true = data.copy()

# generating the missing entries matrix
beta = .25
missing_matrix = np.random.binomial(1, beta, size=data.shape)

# removing entries in data corresponding to missing matrix
for ij, r in np.ndenumerate(missing_matrix):
    if r:
        data[ij] = np.nan


print('Done!')
