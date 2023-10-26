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
        self.current_cash = 100 * 10000
        self.fee = 0.0005
        self.current_ticker = 0

    def check_interval(self, input_interval):
        available_period = ['day', 'minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'minute240', 'week', 'month']
        if input_interval not in available_period:
            print(f"Avaiable interval of data name: {available_period}")
            return sys.exit()
        else:
            return input_interval

    def get_ohlcv(self):
        df = pyupbit.get_ohlcv(ticker=self.ticker, interval=self.interval, count=self.count)
        return df

    def normalize_data(self, df, type='df'):
        #큰 거래량은 작은 거래량보다 훨씬 의미있음.
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        return scaled_data

    def buy_ticker(self, date, rate=1):
        ticker_open_price = self.ohlcv.loc[date][0]
        trade_cash = self.current_cash * rate
        self.current_cash -= trade_cash *
        self.current_ticker += trade_cash / ticker_open_price

    def sell_ticker(self, date, rate=1):
        ticker_open_price = self.ohlcv.loc[date][0]
        trade_ticker = self.current_ticker * rate

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
    def __init__(self, data_count=200, data_interval='minute1', lstm_units=50, dropout_rate=0.2, epochs=20, batch_size=32):
        self.data = Data(count=data_count, interval=data_interval)
        self.ohlcv = self.data.normalize_data(self.data.ohlcv)
        self.model = CustomLSTMModel(lstm_units, dropout_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = 10  # 시퀀스 길이
        self.optimizer = tf.keras.optimizers.Adam()
        self.commission_rate = 0.0005  # 수수료. 변경 가능.

        self.prepare_data()
        self.train()
        self.simulate_trading()

    def prepare_data(self):
        X, Y = [], []
        for i in range(len(self.ohlcv) - self.seq_len - 1):
            X.append(self.ohlcv[i:i+self.seq_len])
            Y.append(self.ohlcv[i+self.seq_len, -2])  # 다음 'close' 값 예측

        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = np.array(X[:split_idx]), np.array(X[split_idx:])
        self.Y_train, self.Y_val = np.array(Y[:split_idx]), np.array(Y[split_idx:])

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_val, self.Y_val))

    def simulate_trading(self):
        # 시작 금액 및 주식 수
        initial_balance = 1000000  # 1,000,000 원
        balance = initial_balance
        current_holding = 0  # 보유 주식 수


        for i in range(len(self.X_val) - 1):
            current_price = self.ohlcv[i][-2]  # 'close' 값
            predicted_next_price = self.model.predict(self.X_val[i].reshape(1, self.seq_len, -1))

            # 투자 상태 출력
            self.print_status(balance, current_holding, current_price)

            # 예측 가격이 현재 가격 + 수수료 이상일 경우 매수
            if predicted_next_price[-1][0] > current_price * (1 + self.commission_rate) and balance > current_price:
                print("Buying all coins at", current_price)
                num_stocks_to_buy = balance // current_price
                balance -= current_price * num_stocks_to_buy * (1 + self.commission_rate)
                current_holding += num_stocks_to_buy

            # 예측 가격이 현재 가격 + 수수료 이하일 경우 매도
            elif predicted_next_price[-1][0] <= current_price * (1 + self.commission_rate) and current_holding > 0:
                print("Selling all coins at", current_price)
                balance += current_holding * current_price * (1 - self.commission_rate)
                current_holding = 0

        # 최종 상태 출력
        self.print_status(balance, current_holding, self.ohlcv[-1][-2])

        final_balance = balance + current_holding * self.ohlcv[-1][-2]

        profit_or_loss = final_balance - initial_balance
        if profit_or_loss > 0:
            print(f"Profit of: {profit_or_loss} KRW")
        elif profit_or_loss < 0:
            print(f"Loss of: {-profit_or_loss} KRW")
        else:
            print("No profit or loss.")

    def print_status(self, balance, current_holding, current_price):
        print(f"Current Total Balance: {balance + current_holding * current_price} KRW")
        print(f"Current Cash Balance: {balance} KRW")
        print(f"Current Stocks: {current_holding}")
        print(f"Current Stocks Value: {current_holding * current_price} KRW")


if __name__ == "__main__":
    vt = VirtualTrading()