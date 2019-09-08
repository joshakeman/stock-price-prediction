import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

df = web.DataReader("AAPL", 'yahoo', start, end)
# print(df.tail())

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
# print(mavg)

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

# compute and show rolling mean/moving average
close_px.plot(label='AAPL')
mavg.plot(label='mavg')
print(plt.legend())
plt.show()

# Compute and show variability/risk
rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')
plt.show()

# Compare competing stocks
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
print(dfcomp)
