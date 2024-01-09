
from main2 import *
class Trader:
    def __init__(self, virtural=True):
        self.is_virtual = virtural
        self.data_num = 5000 #예측에 사용될 데이터 수4
        self.time_step = 100
        self.name = "VirtualTrading"
        self.user = User(name)
        self.ticker = "KRW-BTC"
        self.interval = "minute15"
        self.buy_time = dt.time(9, 00) #오전 9시에 매수
        self.sell_time = dt.time(8, 58) #오전 8시 59분에 판매
        self.predict_time = dt.time(8, 30) #예측시각
        self.current_time = datetime.now()
        if virtural: self.current_time = self.current_time - timedelta(days=10) #1년전 시간부터 출발
        self.df = pyupbit.get_ohlcv(ticker=self.ticker, count=self.data_num, interval=self.interval, to=self.current_time)
        self.target_return = upbit_commission * 2 # 목표 수익률, 업비트 수수료의 두 배

    def manage_class(self):
        if self.is_virtual:
            self.current_time = self.current_time + timedelta(seconds=30)
        else:
            time.sleep(10)
            self.current_time = datetime.now()
        self.add_data()
        #print(f'current time: {self.current_time}')
    def preprocess_data(self, preporcess_df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(preporcess_df)
        return scaled_data, scaler

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.time_step - 1):
            X.append(data[i:(i + self.time_step)])
            Y.append(data[i + self.time_step]) #ohlcv값 예측
        return np.array(X), np.array(Y)

    def add_data(self):
        start_time = self.df.index[-1]
        time_difference = self.current_time - start_time
        if time_difference.total_seconds() > 0:
            number_of_candles = time_difference.total_seconds() / 60 / 15
            df = pyupbit.get_ohlcv(ticker, interval=self.interval, to=self.current_time, count=number_of_candles)
            self.df = pd.concat([self.df, df])

    def is_time_to(self, type):
        if type == "predict":
            check_time = self.predict_time
        elif type == "buy":
            check_time = self.buy_time
        elif type == "sell":
            check_time = self.sell_time
        else: raise "Wrong type in Class Trader, Function is_time_to"
        if self.current_time.hour == check_time.hour and self.current_time.minute == self.predict_time.minute:
            print(f"is time to {type} at {self.current_time}")
            return True
        return False
    def is_time_to_predict(self):
        return self.is_time_to("predict")

    def is_time_to_buy(self):
        return self.is_time_to("buy")

    def is_time_to_sell(self):
        return self.is_time_to("sell")

    def check_buy_or_sell(self, predict):
        predicted = predict[-1][1] - predict[-1][0]
        if predicted >= self.target_return:
            return True
        else:
            return False
    def trade(self):
        while True:
            if self.current_time > datetime.now():
                print("Trading Finish")
                break
            elif self.is_time_to_predict():
                training_df = self.df.tail(200)
                scaled_data, scaler = self.preprocess_data(training_df)
                dataset = scaled_data
                X, Y = self.create_dataset(dataset)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = StockPriceLSTM(input_shape=input_shape, lstm_units=50, stack_depth=2, dropout_rate=0.2)
                model.compile(learning_rate=0.001)
                history = model.train(X_train, y_train, epochs=10, batch_size=100)
                save_history(history, self.current_time, name)
                predictions = model.model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)  # 예측값을 원래 스케일로 변환
                while not self.is_time_to_buy():
                    self.manage_class()
                is_buy = self.check_buy_or_sell(predictions)
                self.user.trade(ticker=ticker, buy=is_buy, trade_time=self.current_time, virtual=self.is_virtual)
            else:
                self.manage_class()
        self.user.asset.save("Test")
        print("*"*50)
        print(self.user.show_portfolio())

if __name__ == '__main__':
    trader = Trader(virtural=False)
    trader.trade()