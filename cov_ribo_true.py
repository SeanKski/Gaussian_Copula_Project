"""
This is to estimate the true covariance of the riboflavin10 data
"""

import numpy as np
from matplotlib import pyplot as plt
import csv

# with open('riboflavinV10.csv', 'r') as file:
#     data = list(csv.reader(file))

# reading in ribo data
data = np.genfromtxt('riboflavinV10.csv', delimiter=',', skip_header=1)
data = np.delete(data, 0, axis=1)  # gets rid of the row title (ribo_X)
# generating cov matrix
cov_true = np.cov(data, rowvar=False)

# generating true correlation matrix
corr_true = np.corrcoef(data, rowvar=True)

print('done')