import numpy as np
import cvxpy as cp
import utils as ut
import resource
from scipy.optimize import minimize, LinearConstraint


def get_instrument_replica(prices, returns, m):
    assert len(prices) == len(returns[0])
    alphas = np.array([i/(m+1) for i in range(1, m+1)])
    t = len(prices)
    mu = returns[:, -1]
    print('number of nans in returns:', np.isnan(returns).sum())
    drawdown = ut.drawdown(prices)
    rhos = np.array([ut.cvar(drawdown, alpha) for alpha in alphas])
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


def forward_portfolio_optimization(returns, alphas, lambdas, n):
    i = len(lambdas)

    def f(y):
        ret = 0.
        for i in range(m):
            ret += lambdas[i] * ut.cvar(ut.drawdown(y@returns), alphas[i])
        return ret

    constraint = LinearConstraint(returns[:, -1], lb=1, ub=np.inf)
    sol = minimize(f, np.ones(n)/n, constraints=constraint)
    print(sol.message)
    return sol.x


if __name__ == "__main__":
    m = 10
    n = 20
    t = 454
    weekly_r0 = np.power(1.03, 1./52)
    r0 = np.array([weekly_r0**i for i in range(t+1)])  # adjusted returns of a risk-free asset
    #prices = ut.get_prices('^GSPC', r0)
    if n == 505:
        returns = ut.get_all_adj_returns(r0)
    else:
        returns = ut.get_adj_returns(n, r0)

    # setting max heap size limit
    rsrc = resource.RLIMIT_DATA
    _, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, ((1024 ** 3) * 8, hard))
    soft, hard = resource.getrlimit(rsrc)
    print('Soft RAM limit set to:', soft / (1024 ** 3), 'GB')

    # forward optimization
    lambdas = np.zeros(m)
    lambdas[3] = 1.
    y_opt = forward_portfolio_optimization(returns, [i/(m+1) for i in range(1, m+1)], lambdas, n)
    print(y_opt)

    # inverse optimization
    prices = np.array(y_opt)@ut.get_adj_returns(n, r0)
    lambdas_opt, v_opt, us_opt, obj_opt = get_instrument_replica(prices, returns, m)
    print('lambdas:', lambdas_opt)
    print('approximation quality:', obj_opt/(obj_opt+v_opt))
