import pyupbit
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense, Add, Dropout

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pyupbit
import sys


class Data:
    def __init__(self, count=200, interval='month'):
        self.ticker = 'KRW-BTC'
        self.count = count
        self.interval = self.check_interval(interval)
        self.ohlcv = self.get_ohlcv()

    def check_interval(self, input_interval):
        available_period = ['day', 'minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60',
                            'minute240', 'week', 'month']
        if input_interval not in available_period:
            print(f"Avaiable interval of data name: {available_period}")
            return sys.exit()
        else:
            return input_interval

    def get_ohlcv(self):
        df = pyupbit.get_ohlcv(ticker=self.ticker, interval=self.interval, count=self.count)
        return df

    def normalize_data(self, df, type='df'):
        # 큰 거래량은 작은 거래량보다 훨씬 의미있음.
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        return scaled_data


class CustomLSTMModel(tf.keras.Model):
    def __init__(self, units, dropout_rate=0.2):
        super(CustomLSTMModel, self).__init__()

        self.lstm1 = LSTM(units, return_sequences=True)
        self.dropout1 = Dropout(dropout_rate)  # 첫 번째 LSTM 레이어 후의 Dropout

        self.lstm2 = LSTM(units, return_sequences=True)
        self.dropout2 = Dropout(dropout_rate)  # 두 번째 LSTM 레이어 후의 Dropout

        self.dropout3 = Dropout(dropout_rate)  # Dense 레이어 전의 마지막 Dropout

        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)

        residual = x
        x = self.lstm2(x)
        x = self.dropout2(x)

        x = Add()([x, residual])

        x = self.dropout3(x)

        return self.dense(x)


class VirtualTrading:
    def __init__(self, data, model, initial_balance=10000000, fee=0.0005):
        self.data = data
        self.model = model
        self.seq_len = 60
        self.fee = fee

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.num_stocks = 0
        self.stock_value = 0

        # Splitting the data
        self.train_size = int(len(data.ohlcv) * 0.8)
        self.X_train = data.ohlcv[:self.train_size]
        self.X_val = data.ohlcv[self.train_size:]

        self.start_trading()

    def start_trading(self):
        for i in range(len(self.X_val) - self.seq_len):
            current_sequence = self.X_val[i:i + self.seq_len]
            current_price = current_sequence[-1, 3]  # getting the close price of the last day
            predicted_next_prices = self.model.predict(current_sequence.reshape(1, self.seq_len, -1))
            predicted_next_price = predicted_next_prices[0, -1]  # getting the last predicted price

            print(
                f"Date: {i + self.seq_len} Current Total Balance: {self.balance + self.num_stocks * current_price:.2f} KRW")
            print(
                f"Current Balance: {self.balance:.2f} KRW, Num of Stocks: {self.num_stocks}, Stock Value: {self.num_stocks * current_price:.2f} KRW")

            if predicted_next_price > current_price + current_price * self.fee:  # buying condition
                if self.balance > 0:
                    self.num_stocks += self.balance / current_price * (1 - self.fee)  # considering the fee
                    self.balance = 0
                    print("Buying stocks")

            elif predicted_next_price < current_price - current_price * self.fee:  # selling condition
                if self.num_stocks > 0:
                    self.balance += self.num_stocks * current_price * (1 - self.fee)  # considering the fee
                    self.num_stocks = 0
                    print("Selling stocks")

            print("-----------------------------")
        """
        # Results
        print(
            f"Initial Total Balance: {self.initial_balance} KRW, Final Total Balance: {self.balance + self.num_stocks * current_price:.2f} KRW")
        
        profit = self.balance + self.num_stocks * current_price - self.initial_balance
        
        if profit > 0:
            print(f"Profit of: {profit:.2f} KRW")
        else:
            print(f"Loss of: {-profit:.2f} KRW")
        """

    def print_status(self, balance, current_holding, current_price):
        print(f"Current Total Balance: {balance + current_holding * current_price} KRW")
        print(f"Current Cash Balance: {balance} KRW")
        print(f"Current Stocks: {current_holding}")
        print(f"Current Stocks Value: {current_holding * current_price} KRW")


if __name__ == "__main__":
    data = Data()
    model = CustomLSTMModel(units=60)
    vt = VirtualTrading(data, model)