import numpy as np
import cvxpy as cp
import utils as ut
#from y_opt import y_opt
import resource
from scipy.optimize import minimize, LinearConstraint


def get_instrument_replica(prices, returns, m, alphas=None):
    assert len(prices) == len(returns[0])
    if alphas is None:
        alphas = np.array([i/m for i in range(1, m+1)])
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


def forward_portfolio_optimization(returns, alphas, lambdas, m, n):

    def f(y):
        ret = 0.
        for i in range(m):
            ret += lambdas[i] * ut.cvar(ut.drawdown(y@returns), alphas[i])
        return ret

    def cons(y):
        return returns[:, -1]@y - 1.

    sol = minimize(f, np.ones(n)/n, constraints={'type': 'eq', 'fun': cons})
    print(sol.message)
    return sol.x


def forward_portfolio_optimization_uncons(returns, alphas, lambdas, m, n):

    def f(y):
        y_ext = np.zeros(len(y)+1)
        y_ext[1:] = y
        y_ext[0] = (1.-returns[1:, -1]@y)/returns[0, -1]
        ret = 0.
        for i in range(m):
            if lambdas[i] > 0:
                ret += lambdas[i] * ut.cvar(ut.drawdown(y_ext@returns), alphas[i])
        return ret

    sol = minimize(f, np.ones(n-1)/n, method='Nelder-Mead')
    print(sol.message)
    y_opt = np.zeros(n)
    y_opt[1:] = sol.x
    y_opt[0] = (1.-returns[1:, -1]@sol.x)/returns[0, -1]
    return y_opt


def forward_portfolio_optimization_maxdd(returns):
    _n, _t = np.shape(returns)
    mdd = cp.Variable()
    y = cp.Variable(_n)
    objective = cp.Minimize(mdd)
    constraints = [y@returns[:, -1] == 1]
    for k in range(_t):
        for i in range(k):
            constraints.append(y@returns[:, i]-y@returns[:, k] <= mdd)
    # optimization
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK)
    return y.value, problem.value


if __name__ == "__main__":
    n = 150
    t = 454
    weekly_r0 = np.power(1.03, 1./52)
    r0 = np.array([weekly_r0**i for i in range(t+1)])  # adjusted returns of a risk-free asset
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
    #lambdas = np.zeros(m)
    #lambdas[-1] = 1.
    #ls = [0., 1.]
    alphas = [1., 0.6, 0.3, 1./t]
    m = len(alphas)
    #alphas = np.array([i/m for i in range(1, m+1)])
    #y_opt = forward_portfolio_optimization_uncons(returns, alphas, ls, m, n)
    y_opt, _ = forward_portfolio_optimization_maxdd(returns)
    y_opt = y_opt[:50]
    returns = returns[:50, :]
    #y_opt = np.array([1.66890243, -0.82224339, -0.25375047])
    #print('optimal y=', y_opt)
    #print('constraint violation=', returns[:, -1]@y_opt-1.)
    #print('cvars for y_opt=', [ut.cvar(ut.drawdown(y_opt@returns), alpha) for alpha in alphas])

    # inverse optimization
    a = 0.8
    #weights = np.random.uniform(0, 1, size=n)
    weights = np.array([1/50]*50)
    weights = weights/(returns[:, -1] @ weights)
    #print('constraint violation=', returns[:, -1] @ weights - 1.)
    #prices = weights@returns
    #print('cvars for weighted=', [ut.cvar(ut.drawdown(weights @ returns), alpha) for alpha in alphas])

    prices = a*y_opt@returns + (1-a)*weights@returns
    #prices = a * y_opt @ returns + (1 - a) * ut.get_prices('^GSPC', r0)
    #print('AvDD=', ut.cvar(ut.drawdown(prices), 1.))
    # avdd = 0.015836979979042803

    #print(alphas)
    #prices = ut.get_prices('^GSPC', r0)
    lambdas_opt, v_opt, us_opt, obj_opt = get_instrument_replica(prices, returns, m, alphas)
    print('lambdas:', lambdas_opt)
    print('approximation quality:', obj_opt/(obj_opt+v_opt))
    print('v_opt=', v_opt)
