import numpy as np
from matplotlib import pyplot as plt
import torch

np.random.RandomState(seed=0)



class Gibbs:
    def __init__(self, filename, beta=0.05):

        self.beta = beta
        self.data, self.data_true, self.data_obs = self.init_data(filename)
        self.cov_true, self.corr_true = self.calc_cov_corr(self.data_true)
        self.cov_obs, self.corr_obs = self.calc_cov_corr(self.data_obs)
#TODO: Calculate the cov and corr difference between the true and observed data to prove the need for a gibbs sampler
#TODO: Implement Gibbs sampler lmao

    def init_data(self, filename):
        """
        Initalizes data: reads data in, and generates a missing matrix with prop beta and creates a matrix of all
        observations without missing data

        :param filename: the name of the file which from to read in data
        """

        # reading in data
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        if np.all(np.isnan(data[:,0])):
            data = np.delete(data, 0, axis=1)  # gets rid of the first column (if it is nan, aka has row titles)
        data_true = data.copy()

        # generating the missing entries matrix
        missing_matrix = np.random.binomial(1, self.beta, size=data.shape)

        # nan'ing entries in data corresponding to missing matrix
        for ij, r in np.ndenumerate(missing_matrix):
            if r:
                data[ij] = np.nan

        # removing any observations which have nan in their samples
        idx = [i for i, v in enumerate(data) if np.any(np.isnan(v))]
        data_obs = np.delete(data, idx, 0) # gets rid of all rows which have nan in them
        return data, data_true, data_obs

    def calc_cov_corr(self, data_matrix):
        cov = np.cov(data_matrix, rowvar=False)
        corr = np.corrcoef(data_matrix, rowvar=False)
        return cov, corr





# Main ##################

if __name__ == '__main__':
    g = Gibbs('riboflavinV10.csv', beta=0.1)
