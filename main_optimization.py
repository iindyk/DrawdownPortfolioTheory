import numpy as np
import cvxpy as cp
import utils as ut
import resource


def get_instrument_replica(prices, m, n, r0):
    alphas = np.array([i/(m+1) for i in range(1, m+1)])
    t = len(prices)
    #returns = ut.get_adj_returns(n, r0)
    returns = ut.get_all_adj_returns(r0)
    assert n == len(returns)
    mu = returns[:, -1]
    print('number of nans in returns:', np.isnan(returns).sum())
    drawdown = ut.drawdown(prices)
    rhos = np.array([ut.cvar(drawdown, alpha) for alpha in alphas])
    print(rhos)
    # create variables
    lambdas = cp.Variable(m)
    v = cp.Variable()
    us = cp.Variable((m, t))
    aux = cp.Variable((m, t))   # aux = max(us, 0)
    objective = cp.Minimize(lambdas@rhos-v)
    constraints = [lambdas >= 0.,
                   cp.sum(lambdas) == 1.,
                   v*mu == returns@cp.sum(us, axis=0),
                   aux >= 0.,
                   aux >= us]

    # add constraints for u and aux
    for i in range(m):
        constraints.append(cp.sum(us[i, :]) == 0.)
        constraints.append(cp.sum(aux[i, :]) <= lambdas[i])
        constraints.append(cp.sum(us[i, :]) >= 0.)
        for k in range(t):
            if k >= 1:
                constraints.append(cp.sum(us[i, :k]) >= 0.)
            constraints.append(us[i, k] >= -1.*lambdas[i]/(alphas[i]*t))

    # optimization
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK)
    return lambdas.value, v.value, us.value, problem.value


if __name__ == "__main__":
    m = 10
    n = 505
    t = 454
    weekly_r0 = np.power(1.03, 1./52)
    r0 = np.array([weekly_r0**i for i in range(t+1)])  # adjusted returns of a risk-free asset
    prices = ut.get_prices('^GSPC', r0)

    # setting max heap size limit
    rsrc = resource.RLIMIT_DATA
    _, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, ((1024 ** 3) * 8, hard))
    soft, hard = resource.getrlimit(rsrc)
    print('Soft RAM limit set to:', soft / (1024 ** 3), 'GB')

    # optimization
    lambdas_opt, v_opt, us_opt, obj_opt = get_instrument_replica(prices, m, n, None)
    print('lambdas:', lambdas_opt)
    print('approximation quality:', obj_opt/(obj_opt+v_opt))
