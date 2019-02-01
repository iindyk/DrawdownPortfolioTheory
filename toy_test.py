import utils as ut


def check_rhos(prices, alphas):
    for alpha in alphas:
        print(ut.cvar(ut.drawdown(prices), alpha))


if __name__ == '__main__':
    p1 = [0., -1., 0.5]
    p2 = [0., -0.5, 0.1]
    alphas = [0.3, 0.6]
    check_rhos(p1, alphas)
    check_rhos(p2, alphas)