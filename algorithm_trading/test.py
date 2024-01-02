import numpy as np
import pyupbit
current_price = pyupbit.get_ohlcv("KRW-BTC", interval='day', count=1, to='2023-01-01')['open']
print(current_price)