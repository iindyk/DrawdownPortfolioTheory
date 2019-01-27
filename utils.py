import pandas as pd
import pickle
import heapq
from operator import itemgetter


def get_adj_returns(n):
    data = pd.read_csv('sp500_joined_closes.csv')
    tickers = get_diverse_list_of_tickers(n)

    # todo
    return None


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
    # todo
    return None


def cvar(alpha, prices):
    # todo
    return None


if __name__ == "__main__":
    print(get_diverse_list_of_tickers(12))