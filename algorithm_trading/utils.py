import pyupbit
import csv
from datetime import datetime, timedelta
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#업비트 수수료
upbit_commission = 0.05 / 100

def fetch_data(ticker, interval, end_date=datetime.now(), count=200):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count, to=end_date)
    return df

def fetch_data_virtual_trading(ticker, interval, start_date, end_date=datetime.now()):
    df_total = pd.DataFrame()

    while start_date < end_date:
        # get_ohlcv를 사용하여 15분봉 데이터 가져오기
        df = pyupbit.get_ohlcv(ticker, interval=interval, to=end_date, count=200)
        if df is not None:
            df_total = pd.concat([df, df_total])
        else:
            break

        # 마지막 데이터의 날짜를 업데이트하여 다음 조회를 위해 설정
        end_date = df.index[0]

        #pyupbit 특성상 빠른 데이터 로드에 의한 오류 방지
        time.sleep(0.1)

    return df_total
ticker = "KRW-BTC"
interval = "minute15"
name = "VirtualTraider"
test_days = 15 # 365일간 예측

#현재 날짜를 기준으로 test_days 까지의 ohlcv 수집
day_ago = datetime.now() - timedelta(days=test_days + 100)
df = fetch_data_virtual_trading(ticker=ticker, interval=interval, start_date=day_ago)

def save_history(history, num, user_name, show=False):
    if show:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    #데이터 csv로 저장
    print(history.history)
    history_df = pd.DataFrame(history.history)
    if not os.path.exists(user_name):
        os.makedirs(user_name)
    filename = f"{user_name}/{num}.csv"
    history_df.to_csv(filename, index=False)

class StockPriceLSTM:
    def __init__(self, input_shape, lstm_units=20, stack_depth=2, dropout_rate=0.2):
        """
        LSTM 모델 초기화
        :param input_shape: 입력 데이터의 형태 (학습 데이터 수)
        :param lstm_units: LSTM 레이어의 유닛 수
        :param stack_depth: LSTM 레이어의 수
        :param dropout_rate: 드롭아웃 비율
        """
        self.model = Sequential()

        # 첫 번째 LSTM 레이어 추가
        self.model.add(LSTM(lstm_units, return_sequences=True if stack_depth > 1 else False,
                            input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))

        # 추가 LSTM 레이어 (스택 깊이에 따라)
        for _ in range(1, stack_depth):
            self.model.add(LSTM(lstm_units, return_sequences=False))
            self.model.add(Dropout(dropout_rate))

        # 출력 레이어
        self.model.add(Dense(2))

    def compile(self, learning_rate=0.001):
        """
        모델 컴파일
        :param learning_rate: 학습률
        """
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=500, batch_size=64, validation_split=0.2, steps_per_epoch=200,
              validation_steps=50):
        """
        모델 훈련
        :param X_train: 훈련 데이터
        :param y_train: 타깃 데이터
        :param epochs: 에폭 수
        :param batch_size: 배치 크기
        :param validation_split: 검증 데이터 비율
        :param steps_per_epoch: 에폭 당 스텝 수
        :param validation_steps: 검증 당 스텝 수
        """
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                              validation_split=validation_split, steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps)


if __name__ == '__main__':
    pass
    #model = StockPriceLSTM(input_shape=(60, 1))
    #model.compile()



