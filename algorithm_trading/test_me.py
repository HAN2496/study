import pandas as pd
import pyupbit
df_total = pd.DataFrame()


df = pyupbit.get_ohlcv("KRW-BTC", count=1, interval="day", to="20201010")
# 마지막 200개의 행을 추출
last_200_rows = df.tail(200)

print(df['close'])
