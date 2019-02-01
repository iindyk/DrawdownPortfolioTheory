import utils as ut
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog


def check_rhos(prices, alphas):
    for alpha in alphas:
        print(ut.cvar(ut.drawdown(prices), alpha))


def optimal_portfolio_of2(prices1, prices2, alphas, lambdas):
    assert len(lambdas) == len(alphas)
    m = len(lambdas)

    def f(y):
        ret = 0.
        for i in range(m):
            ret += lambdas[i]*ut.cvar(ut.drawdown(y[0]*prices1+y[1]*prices2), alphas[i])
        return ret

    constraint = LinearConstraint(np.array([prices1[-1], prices2[-1]]), lb=.99999, ub=np.inf)
    sol = minimize(f, np.array([1., 1.]), constraints=constraint)
    print(sol.message)
    return sol.x


def inverse_optimization():
    x0 = np.zeros(15)
    # x[:3]=u^1, x[3:6]=u^2, x[6:9]=zeta^1, x[9:12]=zeta^2, , x[12]=l_1, x[13]=l_2, x[14]=v

    # cost vector
    c = np.zeros_like(x0)
    c[12] = 1.674
    c[13] = 2.242
    c[14] = -1

    # equality constraints
    A_eq = np.zeros((5, 15))
    b_eq = np.zeros(5)

    A_eq[0, 12] = 1
    A_eq[0, 13] = 1
    b_eq[0] = 1

    A_eq[1, 0], A_eq[1, 1], A_eq[1, 2] = 4, 4, 1
    A_eq[1, 3], A_eq[1, 4], A_eq[1, 5] = 4, 4, 1
    A_eq[1, 14] = -1

    A_eq[2, 0], A_eq[2, 1], A_eq[2, 2] = 3, 1, 1
    A_eq[2, 3], A_eq[2, 4], A_eq[2, 5] = 3, 1, 1
    A_eq[2, 14] = -1

    A_eq[3, 0], A_eq[3, 1], A_eq[3, 2] = 1, 1, 1

    A_eq[4, 3], A_eq[4, 4], A_eq[4, 5] = 1, 1, 1

    # inequality constraints
    A_ub = np.zeros((16, 15))
    b_ub = np.zeros(16)

    A_ub[0, 0], A_ub[0, 1] = -1, -1

    A_ub[1, 3], A_ub[1, 4] = -1, -1

    A_ub[2, 0], A_ub[2, 12] = -1, -1 / (3 * 0.3)
    A_ub[3, 1], A_ub[3, 12] = -1, -1 / (3 * 0.3)
    A_ub[4, 2], A_ub[4, 12] = -1, -1 / (3 * 0.3)

    A_ub[5, 3], A_ub[5, 13] = -1, -1 / (3 * 0.6)
    A_ub[6, 4], A_ub[6, 13] = -1, -1 / (3 * 0.6)
    A_ub[7, 5], A_ub[7, 13] = -1, -1 / (3 * 0.6)

    A_ub[8, 0], A_ub[8, 6] = 1, -1
    A_ub[9, 1], A_ub[9, 7] = 1, -1
    A_ub[10, 2], A_ub[10, 8] = 1, -1
    A_ub[11, 3], A_ub[11, 9] = 1, -1
    A_ub[12, 4], A_ub[12, 10] = 1, -1
    A_ub[13, 5], A_ub[13, 11] = 1, -1

    A_ub[14, 6], A_ub[14, 7], A_ub[14, 8], A_ub[14, 12] = 1, 1, 1, -1

    A_ub[15, 9], A_ub[15, 10], A_ub[15, 11], A_ub[15, 13] = 1, 1, 1, -1

    # bounds
    bounds = [(0.0, None), (None, None), (None, None),
              (0.0, None), (None, None), (None, None),
              (0.0, None), (0.0, None), (0.0, None),
              (0.0, None), (0.0, None), (0.0, None),
              (0.0, None), (0.0, None),
              (None, None)]
    sol = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
    print(sol.message)
    print('lambdas=', sol.x[12:14])
    print('v=', sol.x[14])
    print('objective=', sol.fun)


if __name__ == '__main__':
    p1 = np.array([4., 4., 1.])
    p2 = np.array([3., 1., 1.])
    alphs = [0.3, 0.6]
    ls = [0, 1]
    #check_rhos(p1, alphs)
    #check_rhos(p2, alphs)
    #print(optimal_portfolio_of2(p1, p2, alphs, ls))
    #optimal_weights = [0.484, 0.516]
    #p = optimal_weights[0]*p1+optimal_weights[1]*p2
    #check_rhos(p, alphs)
    inverse_optimization()