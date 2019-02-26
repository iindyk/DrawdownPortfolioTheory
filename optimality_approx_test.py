import numpy as np
from optimality_approximation import get_constraint_violation
from main_optimization import forward_portfolio_optimization_uncons


returns = np.array([[4, 4, 1], [3, 1, 1]])
alphas = [1]
lambdas = [1.]
m = 1
n = 2
y_opt = forward_portfolio_optimization_uncons(returns, alphas, lambdas, m, n)
print(y_opt)
opt_ret = y_opt@returns
print('constraint violation is', get_constraint_violation(opt_ret, returns, alphas[0]))