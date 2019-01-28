import numpy as np
import cvxpy as cp
import utils as ut


def get_instrument_replica(prices, m, n):
    alphas = np.array([i/(m+1) for i in range(1, m+1)])
    returns = ut.get_adj_returns(n)
    _, t = np.shape(returns)
    mu = returns[:, -1]
    rhos = np.array([ut.cvar(ut.drawdown(prices), alpha) for alpha in alphas])

    # create variables
    lambdas = cp.Variable(m)
    v = cp.Variable(1)
    us = cp.Variable((m, t))
    aux = cp.Variable((m, t))   # aux = max(us, 0)
    objective = cp.Minimize(cp.multiply(lambdas, rhos)-v)
    constraints = [lambdas >= 0.,
                   cp.sum(lambdas) == 1.,
                   v*mu == np.multiply(returns, cp.sum(us, axis=0)),
                   aux >= 0.,
                   aux >= us]

    # add constraints for u
    for i in range(m):
        constraints.append(cp.sum(us[i]) == 0.)
        constraints.append(cp.sum(aux[i]) <= lambdas[i])
        for k in range(t):
            constraints.append(cp.sum(us[i, :k]) >= 0.)
            constraints.append(us[i, k] >= -1.*lambdas[i]/(alphas[i]*t))

    # optimization
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return lambdas.value, v.value, us.value, objective.value


if __name__ == "__main__":
    m = 20
    n = 20
    prices = ut.get_prices('S&P500')
    print(get_instrument_replica(prices, m, n))
