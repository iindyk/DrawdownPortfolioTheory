import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
import pandas as pd


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        tickers.append({'ticker': row.findAll('td')[0].text,
                        'name': row.findAll('td')[1].text,
                        'sector': row.findAll('td')[3].text.replace('\n', '')})
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


def save_market_cap_data():
    f = open("sp500tickers.pickle", "rb")
    tickers = pickle.load(f)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    for ticker in tickers:
        r = requests.get("http://www.zacks.com/stock/quote/"+ticker['ticker'], headers=headers)
        soup = bs.BeautifulSoup(r.content, "html.parser")
        for tr in soup.findAll("table", class_="abut_bottom"):
            for td in tr.find_all("td"):
                if td.text == "Market Cap":
                    print(td.find_next_sibling("td").text)
                    ticker['market_cap'] = td.find_next_sibling("td").text
    f.close()
    f = open("sp500tickers.pickle", "wb")
    pickle.dump(tickers, f)


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker['ticker'])):
            df = web.DataReader(ticker['ticker'], 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker['ticker']))
        else:
            print('Already have {}'.format(ticker['ticker']))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    # add indices
    tickers.append({'ticker': '^GSPC'})
    tickers.append({'ticker': '^IXIC'})
    tickers.append({'ticker': '^SML'})
    tickers.append({'ticker': '^DJI'})

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker['ticker']))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker['ticker']}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('joined_closes.csv')


if __name__ == "__main__":
    compile_data()
