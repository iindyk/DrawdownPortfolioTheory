import pandas as pd
import pickle


def get_adj_returns(n):
    data = pd.read_csv('sp500_joined_closes.csv')
    # todo
    return None


def get_diverse_list_of_tickers(n):
    f = open("sp500tickers.pickle", "rb")
    tickers = pickle.load(f)
    sectors = set()
    for ticker in tickers:
        sectors.add(ticker['sector'])
    print(sectors)
    f.close()


def get_prices(name):
    # todo
    return None


def cvar(alpha, prices):
    # todo
    return None


if __name__ == "__main__":
    print(get_diverse_list_of_tickers(100))