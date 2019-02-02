import pandas as pd
import pickle
import heapq
import numpy as np
from scipy.optimize import minimize
from operator import itemgetter


def get_adj_returns(n, r0):
    data = pd.read_csv('joined_closes_weekly.csv')
    tickers = get_diverse_list_of_tickers(n)
    prices = []
    for ticker in tickers:
        prices.append(list(data[ticker]))
    prices = np.array(prices)
    t = len(prices[0])
    adj_returns = np.zeros_like(prices)
    for i in range(n):
        for j in range(1, t):
            adj_returns[i, j] = prices[i, j]/(r0[j]*prices[i, 0])-1.
    return adj_returns[:, 1:]


def get_diverse_list_of_tickers(n):
    f = open("sp500tickers.pickle", "rb")
    tickers = pickle.load(f)
    f.close()
    sectors = set()
    for ticker in tickers:
        sectors.add(ticker['sector'])
    n_sectors = len(sectors)

    # split n on sectors
    partition = [int(n/n_sectors)]*(n_sectors-1)
    partition.append(n-int(n/n_sectors)*(n_sectors-1))
    return_list = []
    market_cap_na = 0
    for k, sector in zip(partition, sectors):
        # get companies within this sector
        tmp_sector = {}
        for ticker in tickers:
            if ticker['sector'] == sector:
                try:
                    tmp_sector[ticker['ticker']] = float(ticker['market_cap'].replace(' B', ''))
                except KeyError:
                    market_cap_na += 1
        # get n companies with the biggest market cap from sector
        return_list.extend(dict(heapq.nlargest(k, tmp_sector.items(), key=itemgetter(1))).keys())
    f.close()
    print('Market Cap data was not available for', market_cap_na, 'companies')
    return return_list


def get_prices(name, r0):
    assert name in ['^GSPC', '^IXIC', '^SML', '^DJI', 'A&M']
    # read data from file
    data = pd.read_csv('joined_closes_weekly.csv')
    if name != 'A&M':
        prices = list(data[name])
    else:
        prices = np.array(list(data['AMZN']))+np.array(list(data['MSFT']))
    t = len(prices)
    adj_returns = np.zeros(t)
    for i in range(t):
        adj_returns[i] = prices[i]/(r0[i]*prices[0])-1.
    return adj_returns[1:]


def cvar(losses, alpha):
    # todo: fix
    l = len(losses)

    def f(c):
        return c + (1. / ((alpha)*l)) * np.sum([max(p - c, 0) for p in losses])
    sol = minimize(f, np.array(0.0))
    try:
        return sol.fun[0]
    except TypeError:
        return sol.fun


def drawdown(prices):
    n = len(prices)
    d = np.zeros(n)
    for i in range(n):
        d[i] = max(prices[:i+1])-prices[i]
    return d


if __name__ == "__main__":
    prices = np.array([-100, -20, -20, -20, 0, 0, 0, 0, 50, 50])
    losses = -1*prices
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
    for alpha in alphas:
        print(cvar(losses, alpha))