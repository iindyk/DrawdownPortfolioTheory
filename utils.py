import pandas as pd
import pickle
import heapq
import numpy as np
from operator import itemgetter


def get_adj_returns(n):
    data = pd.read_csv('joined_closes.csv')
    tickers = get_diverse_list_of_tickers(n)
    prices = []
    for ticker in tickers:
        prices.append(list(data[ticker]))
    prices = np.array(prices)
    t = len(prices[0])
    r0 = np.ones(t)   # todo: return of a risk-free asset
    adj_returns = np.zeros_like(prices)
    for i in range(n):
        for j in range(t):
            adj_returns[i, j] = prices[i, j]/(r0[j]*prices[i, 0])-1.
    return adj_returns


def get_diverse_list_of_tickers(n):
    f = open("sp500tickers.pickle", "rb")
    tickers = pickle.load(f)
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


def get_prices(name):
    assert name in ['^GSPC', '^IXIC', '^SML', '^DJI']
    # read data from file
    data = pd.read_csv('joined_closes.csv')
    return list(data[name])


def cvar(alpha, prices):
    # todo
    return None


def drawdown(prices):
    # todo
    return None


if __name__ == "__main__":
    print(get_adj_returns(12))