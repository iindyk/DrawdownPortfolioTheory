import numpy as np
import utils as ut
from main_optimization import *


m = 2
n = 2
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

ls = [0., 1.]
alphas = [1., 1./t]
y_opt1 = forward_portfolio_optimization_uncons(returns, alphas, ls, m, n)
print('optimal y1=', y_opt1)
print('constraint violation=', returns[:, -1]@y_opt1-1.)
print('MaxDD value=', max(ut.drawdown(y_opt1@returns)))

y_opt2, opt_val = forward_portfolio_optimization_maxdd(returns)
print('optimal y2=', y_opt2)
print('constraint violation=', returns[:, -1]@y_opt2-1.)
print('MaxDD value=', max(ut.drawdown(y_opt2@returns)))
print('Problem optimal value=', opt_val)
