import pyupbit
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker, interval, count):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df

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
        self.model.add(Dense(1))

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



model = StockPriceLSTM(input_shape=(60, 1))
model.compile()



