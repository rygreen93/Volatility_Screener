

from pybit import usdt_perpetual
import numpy as np
import time
import pandas as pd
import itertools
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import os
import ccxt
import plotly.express as px
import chart_studio.plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import chart_studio.plotly as py
from talib import TRANGE, ATR,EMA 

#Script Section Selects Coins Below Min Buy Size Specified

min_notional = float(input("Enter Maximum Notional Size To Filter in $?"))
min_cap = float(input("Enter Maximum Market Cap Position (CMC)?"))

session_unauth = usdt_perpetual.HTTP(
    endpoint="https://api-testnet.bybit.com"
)


print ("Loading Bybit Markets")
markets = session_unauth.query_symbol()
#print (markets)

for key in markets.keys():
    print (key)

markets = markets['result']
##print (markets)

usdt = list(filter(lambda thing: thing.get('quote_currency') == 'USDT', markets))

symbols = [i['name'] for i in usdt]
symbols_name = [i['base_currency'] for i in usdt]
lot = [d['lot_size_filter'] for d in usdt]
min_qty = [i['min_trading_qty'] for i in lot]


# add USDT to symbols
currency = "/USDT:USDT"
symbols_name = [i['base_currency'] for i in usdt]
symbols_usdt = ["{}{}". format(i,currency) for i in symbols_name]


price = []
for x in range(len(symbols)):
    val = session_unauth.public_trading_records(
    symbol=symbols[x],
    limit=1)
    price.append(val)


c_price = [i['result'] for i in price]

k_price = list(itertools.chain.from_iterable(c_price))
k_price = [d['price'] for d in k_price]

df = pd.DataFrame()
df['symbols'] = symbols_usdt
df['min_lot'] = min_qty
df['price'] = k_price
df['min_buy'] = df['min_lot'] * df['price']

#enter min_buy size that you would like to filter out
df = df[df['min_buy'] < min_notional]

##df_sorted = df.sort_values(['min_buy', 'price', 'min_lot', 'symbols'], ascending = True)
##print(df_sorted)
##df_sorted.to_csv('one.csv')

select_symbols = df['symbols'].tolist()

for elem in list(select_symbols):
    if elem == "XNO/USDT:USDT" or elem == "10000NFT/USDT:USDT" or elem=="USDC/USDT:USDT":
        select_symbols.remove(elem)
        
print("Coins Selected Based on Buy Size:", select_symbols)

########## market cap Part
# find the market cap
print('Filtering by marketCap !')
headers = {'X-CMC_PRO_API_KEY': 'Your CMC API Key'}
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
r = requests.get(url, headers=headers)
find_ranks = {}
if r.status_code == 200:
    data = r.json()
    #print(data)
    for d in data['data']:
        symbol = d['symbol']
        find_ranks[symbol] = d['cmc_rank']

after_market_cap=[]
for symbol in select_symbols:
    marketcap_symbol = symbol.upper().replace('/USDT:USDT', '')
    marketcapPosition = find_ranks[marketcap_symbol] if (marketcap_symbol in find_ranks) else 999

    if (marketcapPosition > min_cap) :
        continue
    after_market_cap.append(symbol)

    print('Coin accepted :', symbol, ' / marketcap position : ', marketcapPosition)
 


print ("Symbol List: ", after_market_cap)
time.sleep(5)

ftx = ccxt.ftx({
    'apiKey': '',
    'secret': '',
    'rateLimit': 200,
    'enableRateLimit': True
})

bybit = ccxt.bybit({
    'apiKey': 'Your API',
    'secret': 'Your Key',
    'rateLimit': 200,
    'enableRateLimit': True
})

markets = bybit.load_markets()

symbols = after_market_cap        
msec = 1000
minute = 60 * msec
hold = 30

from_datetime = str(datetime.now()-relativedelta(months=+1))
#from_datetime = '2022-06-01 00:00:00'
from_timestamp = bybit.parse8601(from_datetime)
now = bybit.milliseconds()
#from_datetime = '2021-12-01 00:00:00'
#from_datetime = '2021-12-01 00:00:00'
#to_datetime = '2021-12-31 23:59:59'
#from_timestamp = ftx.parse8601(from_datetime)
#now = ftx.parse8601(to_datetime)

df = pd.DataFrame()
def get_candles(_symbol,_start_timestamp,_end_timestamp):
    _data = []
    while _start_timestamp < _end_timestamp:
    
        try:

            print(bybit.milliseconds(), 'Fetching candles starting from', bybit.iso8601(_start_timestamp))
            ohlcvs = bybit.fetch_ohlcv(_symbol, '1m', _start_timestamp)
            print(bybit.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
            first = ohlcvs[0][0]
            last = ohlcvs[-1][0]
            print('First candle epoch', first, bybit.iso8601(first))
            print('Last candle epoch', last, bybit.iso8601(last))
            _start_timestamp += len(ohlcvs) * minute
            _data += ohlcvs
            

        except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:

            print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
            time.sleep(hold)
    return _data

data={'Symbol':[],'Volatility':[],'Variance %':[],'Volume':[],'Linear Trend %':[],'1% Candles':[],'Potato Indicator':[]}
metrics = pd.DataFrame(data)

atr_multiplier = 5

for symbol in symbols:
    print('Fetching Candles for ',symbol)
    try:
        temp = get_candles(symbol,from_timestamp,now)
        df = pd.DataFrame(temp, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['Volatility %']=(df['high']-df['low'])/df['close']*100
        print ('Volatility %')
        df['Normalized Close']=(df['close']-df['close'].mean())/df['close'].std()
        coefficients, residuals, _, _, _ = np.polyfit(range(len(df.index)),df['Normalized Close'],1,full=True)
        mse = residuals[0]/(len(df.index))
        nrmse = np.sqrt(mse)/(df['Normalized Close'].max() - df['Normalized Close'].min())
        print('Slope ' + str(coefficients[0]))
        print('NRMSE: ' + str(nrmse))
        one_pct_candles = df[df['Volatility %']>1]['Volatility %'].count()
        #Calculate ATR channels
        low=df['low'].to_numpy()
        close=df['close'].to_numpy()
        high=df['high'].to_numpy()
        volume=df['volume'].to_numpy()
        tr = TRANGE(high,low,close)
        atr_14 = ATR(high, low, close, timeperiod=14)
        ema_14 = EMA(close, timeperiod=14)
        
        volatility_shift = atr_multiplier*atr_14
        channel_above = ema_14+volatility_shift
        channel_below = ema_14-volatility_shift
        df['ATR4_14_UP'] = channel_above
        df['ATR4_14_DOWN'] = channel_below
        ATR_14_BREAK = df[df['close']>df['ATR4_14_UP']]['time'].count()+df[df['close']<df['ATR4_14_DOWN']]['time'].count()
        if (ATR_14_BREAK > 20000):
            ATR_14_BREAK = 0
    
        if symbol == symbols[0]:
            data={'Symbol':[symbol],'Volatility':[df.mean()['Volatility %']],'Variance %':[df.var()['close']/df.mean()['close']*100],'Volume':[df.mean()['volume']],'Linear Trend %':[coefficients[0]*100],'1% Candles':[one_pct_candles],'Potato Indicator':[ATR_14_BREAK]}
            metrics = pd.DataFrame(data)
            print(data)
        else:
            data={'Symbol':symbol,'Volatility':df.mean()['Volatility %'],'Variance %':df.var()['close']/df.mean()['close']*100,'Volume':df.mean()['volume'],'Linear Trend %':coefficients[0]*100,'1% Candles':one_pct_candles,'Potato Indicator':ATR_14_BREAK}
            metrics = metrics.append(data,ignore_index=True)
            print(data)
    except:
        print("An exception occurred on symbol ",symbol)




Title_1 = "Coin Volatility vs Variance % Volume and Trend from "+str(from_datetime)+" to "+str(datetime.now())
fig = px.scatter(metrics, x="Variance %", y="Volatility",
	         size="Volume", color="Linear Trend %", hover_name="Symbol", log_x=True, size_max=80,template="simple_white", title=Title_1)

fig.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.write_html("/root/screener/bybit_volatility.html")
#fig.write_html("/home/dario/nextcloud/crypto_metrics/volatility.html")
fig.show()
#py.plot(fig, filename = 'Equilibrium Variance % Volatility %', auto_open=True)




Title_2 = "Coin  >1pct candle vs Variance % Volume and Trend from "+str(from_datetime)+" to "+str(datetime.now())
fig = px.scatter(metrics, x="Variance %", y="1% Candles",
	         size="Volume", color="Linear Trend %", hover_name="Symbol", log_x=True, log_y=True, size_max=60,template="simple_white", title=Title_2)

fig.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.write_html("/root/screener/bybit_candles.html")
fig.show()
#py.plot(fig, filename = 'Equilibrium Variance % 1% Candles', auto_open=True)

Title_3 = "Potato indicator (close brakes "+str(atr_multiplier)+"*ATR14 Channel) vs Variance % Volume and Trend from "+str(from_datetime)+" to "+str(datetime.now())
fig = px.scatter(metrics, x="Variance %", y="Potato Indicator",
	         size="Volume", color="Linear Trend %", hover_name="Symbol", log_x=True, log_y=True,size_max=60,template="simple_white", title=Title_3)

fig.update_xaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showgrid=True, showline=True, linewidth=2, linecolor='black', mirror=True)
fig.write_html("/root/screener/bybit_potato.html")
fig.show()
#py.plot(fig, filename = 'Potato Indicator', auto_open=True)



