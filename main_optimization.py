import numpy as np
import cvxpy as cvx
import utils as ut


def get_instrument_replica(prices, m, n, t):
    alphas = np.array([i/(m+1) for i in range(1, m+1)])
    returns = ut.get_adj_returns(n)
    mu = np.array([0.5]*n)