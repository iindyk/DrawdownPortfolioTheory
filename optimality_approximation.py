import cvxpy as cp
import utils as ut
import numpy as np
import resource


def get_constraint_violation(optimal_returns, all_returns, alpha):
    n, t = np.shape(all_returns)

    # calculate CDaR
    drawdowns = ut.drawdown(optimal_returns)
    q_var = cp.Variable(t)
    objective = cp.Maximize(q_var@drawdowns)
    constraints = [q_var >= 0,
                   q_var <= 1./(alpha*t),
                   cp.sum(q_var) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK)
    cdar = problem.value
    q_opt = q_var.value
    betas = np.zeros(n)
    print('cdar1=', cdar, 'cdar2=', ut.cvar(drawdowns, alpha))

    # construct list of indices of max elements
    k = []
    for j in range(1, t+1):
        k.append(np.argmax(optimal_returns[:j]))

    # construct betas
    for i in range(n):
        for j in range(t):
            betas[i] += q_opt[j]*(all_returns[i, k[j]]-all_returns[i, j])/cdar

    return np.linalg.norm(all_returns[:, -1]-optimal_returns[-1]*betas)


if __name__ == '__main__':
    n = 100
    t = 454
    weekly_r0 = np.power(1.03, 1. / 52)
    r0 = np.array([weekly_r0 ** i for i in range(t + 1)])  # adjusted returns of a risk-free asset
    if n == 505:
        returns = ut.get_all_adj_returns(r0)
    else:
        returns = ut.get_adj_returns(n, r0)
    prices = ut.get_prices('^GSPC', r0)
    alphas = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1./t]

    # setting max heap size limit
    rsrc = resource.RLIMIT_DATA
    _, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, ((1024 ** 3) * 8, hard))
    soft, hard = resource.getrlimit(rsrc)
    print('Soft RAM limit set to:', soft / (1024 ** 3), 'GB')

    for a in alphas:
        print('for alpha=', a, 'constraint violation=', get_constraint_violation(prices, returns, a))



