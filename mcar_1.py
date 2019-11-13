import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy import stats





class Gibbs:
    def __init__(self, filename, beta=0.05, seed=0):
        """

        :param filename: The name/path of the file from which to read data
                NOTE: The data must have the parameters in the columns and the observations as rows
        :param beta: the probability of missing values
        data_obs = the data matrix with nans at missing value points
        data_mis = the data matrix with nans at observed value points
        data = the data with all true values
        """
        self.seed = seed
        self.beta = beta
        self.data_true, self.data_obs, self.data_mis, self.r = self.init_data(filename)

        self.cov_true, self.corr_true = self.calc_cov_corr(self.data_true)

        self.cov_obs, self.corr_obs = self.calc_cov_corr(self.data_obs)

        self.z_obs = self.calc_z_obs(self.data_obs, self.r)


    def init_data(self, filename):
        """
        Initalizes data: reads data in, and generates a missing matrix with prop beta and creates a matrix of all
        observations without missing data

        :param filename: the name of the file which from to read in data
        """

        # reading in data
        data_true = np.genfromtxt(filename, delimiter=',', skip_header=1)
        if np.all(np.isnan(data_true[:,0])):
            data_true = np.delete(data_true, 0, axis=1)  # gets rid of the first column if nan, aka has row titles

        #######FOR TESTING
#TODO: Delete this line:
        data_true = data_true[:15,:]
        ###################

        data_obs = np.full_like(data_true, np.nan)
        data_mis = data_obs.copy()

        # generating the missing entries matrix
        np.random.seed(self.seed)
        observed_matrix = np.random.binomial(1, 1 - self.beta, size=data_true.shape)  # rij = 1 means value is observed

        # nan'ing entries in data corresponding to missing matrix
        for ij, r in np.ndenumerate(observed_matrix):
            if r:
                data_obs[ij] = data_true[ij]
            else:
                data_mis[ij] = data_true[ij]

        # # removing any observations which have nan in their samples
        # idx = [i for i, v in enumerate(data_obs) if np.any(np.isnan(v))]
        # data_obs_nan_rows_removed = np.delete(data_obs, idx, 0) # gets rid of all rows which have nan in them
        return data_true, data_obs, data_mis, observed_matrix

    def calc_cov_corr(self, data_matrix):
        if not np.any(np.isnan(data_matrix)):
            cov = np.cov(data_matrix, rowvar=False)
        else:
            cov = np.full((len(data_matrix[0]), len(data_matrix[0])), np.nan)
        corr = self.calc_corr(data_matrix)
        return cov, corr

    def calc_corr(self, data_matrix):
        p = len(data_matrix.T)  # data_matrix is size (n,p)
        corr = np.zeros((p, p))
        for j in range(p):
            for k in range(p):
                corr[j, k] = stats.kendalltau(data_matrix[:, j], data_matrix[:, k], nan_policy='omit')[0]
        return corr

    def gibbs_step_1(self, z_cal, z_corr):
        c = z_corr
        r = self.r
        for j, col in enumerate(z_cal.T):
            nj = list(range(len(c.T)))
            del nj[j]

            c_nj_nj = c[np.ix_(nj, nj)]
            c_j_nj = c[np.ix_([j], nj)]
            v = np.matmul(c_j_nj, np.linalg.inv(c_nj_nj)).T  # v_T = C[j,-j]*((C[-j,-j])^-1)
            sig_sqd = c[j, j] - np.matmul(v.T, c[np.ix_(nj, [j])])

            for i in range(len(r)):
                if not r[i, j]:
                    if np.any(np.isnan(z_calc)): # if there are nans then this is the first round of sampling
                        # seeing if any other elements in row i are nan
                        nans_in_row = np.where(np.isnan(z_cal[i]))[0]
                        row_no_nan = np.delete(z_cal[i], nans_in_row)
                        if len(nans_in_row) > 1:  # more than one nan in row
                            v_no_corresponding_nan = np.delete(v.T, np.where(nans_in_row != j))  # take out locations \
                            #  in v corresponding to nans in row i, OTHER THAN the nan in i,j (the nan we are imputing)
                        else:
                            v_no_corresponding_nan = v
                        mu = np.matmul(row_no_nan, v_no_corresponding_nan)

                    else:  # this is not the first round of sampling, thus no nans
                        mu = np.matmul(z_calc[np.ix_([i], nj)], v)  # much easier eh?

                    z_cal[i, j] = np.random.normal(mu, sig_sqd)

        return z_cal

    def gibbs_step_2(self, z_obs):
        n = len(z_obs.T)
        scale_prior = np.identity(n)
        DOF_priof = 1
        wishart = stats.invwishart(DOF_priof+n, scale_prior + np.matmul(z_obs.T,z_obs))
        corr = wishart.rvs(random_state=0)

        return corr

    def calc_z_obs(self, data_obs, r):
        # setting up the empirical cumulative distribution function
        z_obs = np.full_like(data_obs, np.nan)
        for j, (col, r_col) in enumerate(zip(data_obs.T, r.T)):
            for i, data_ij in enumerate(col):
                top = np.sum(r_col * (col < data_ij)) + 1  # top = sum(rdj * indicator(ydj < yij)) + 1
                bot = np.sum(r_col) + 1  # sum(number of data in col_j) + 1
                if r_col[i]:  # if observed
                    z_obs[i,j] = stats.norm.ppf(top / bot)  # z_obs[ij] = NormalCDF_inv(top / bot)
        return z_obs






# Main ##################

if __name__ == '__main__':
    g = Gibbs('riboflavinV10.csv', beta=0.1)

    for gen in range(100):
        if gen == 0:
            #initilze priors
            corr_calc = np.identity(len(g.data_true.T))
            z_calc = g.z_obs

        z_calc = g.gibbs_step_1(z_calc, corr_calc)
        corr_calc = g.gibbs_step_2(z_calc)


    # calculating the L1 norms of the covariance and corr data
    corr_l1_norm = np.linalg.norm(g.corr_true - g.corr_obs, ord=1, axis=1)
    cov_l1_norm = np.linalg.norm(g.cov_true - g.corr_obs, ord=1, axis=1)

    print('Done!')
