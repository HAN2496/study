import pyupbit

from utils import *
#from utils import StockPriceLSTM

class Asset:
    def __init__(self):
        self.portfolio = {}  # 형식: {ticker: [수량, 가격]}
        self.cash = 1000000000
        self.total_asset = self.cash
        self.start_cash = self.cash
        self.transaction_history = []  # 거래 기록

    def add_cash(self, amount):
        self.cash += amount
        self.update_total_asset()

    def buy(self, ticker, quantity, price, buy_time):
        cost = quantity * price
        if cost > self.cash:
            print("Insufficient cash to buy.")
            return False
        self.cash -= cost
        if ticker in self.portfolio:
            self.portfolio[ticker][0] += quantity
        else:
            self.portfolio[ticker] = [quantity, price]
        self.update_total_asset()
        self.record_transaction('buy', ticker, quantity, price, buy_time)
        return True

    def sell(self, ticker, quantity, price, sell_time):
        if ticker not in self.portfolio or self.portfolio[ticker][0] < quantity:
            print("Insufficient asset to sell.")
            return False
        self.portfolio[ticker][0] -= quantity
        self.cash += quantity * price
        if self.portfolio[ticker][0] == 0:
            del self.portfolio[ticker]
        self.update_total_asset()
        self.record_transaction('sell', ticker, quantity, price, sell_time)
        return True

    def update_total_asset(self):
        self.total_asset = self.cash + sum(quantity * price for quantity, price in self.portfolio.values())

    def record_transaction(self, transaction_type, ticker, quantity, price, record_time):
        self.transaction_history.append({
            "type": transaction_type,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "timestamp": record_time
        })

    def show_portfolio(self):
        print("Current Portfolio:")
        for ticker, (quantity, price) in self.portfolio.items():
            print(f"{ticker}: {quantity} units at {price} each")
        print(f"Total Asset: {self.total_asset}")
        print(f"Cash: {self.cash}")
        print(f"Start Cash: {self.start_cash}")
        print(f"Earn: {(self.start_cash - self.cash)/self.start_cash * 100} (%)")
        print(f"*"*30)

    def save(self, transactions_filename):
        transactions_filename = f"data/{transactions_filename}"
        # 거래 기록을 CSV 파일로 저장
        with open(transactions_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type', 'Ticker', 'Quantity', 'Price', 'Timestamp'])
            for transaction in self.transaction_history:
                writer.writerow([transaction['type'], transaction['ticker'],
                                 transaction['quantity'], transaction['price'],
                                 transaction['timestamp']])

class User:
    def __init__(self, name):
        self.name = name
        self.asset = Asset()
        self.history_num = 60
        self.history_candlestick = "minute15"
        self.lstm_model = StockPriceLSTM((self.history_num, 1))
        self.target_return = upbit_commission * 2 # 업비트 수수료의 두 배

    def add_cash(self, amount):
        self.asset.add_cash(amount)

    def buy_crypto(self, ticker, quantity, price, trade_time):
        return self.asset.buy(ticker, quantity, price, trade_time)

    def sell_crypto(self, ticker, quantity, price, trade_time):
        return self.asset.sell(ticker, quantity, price, trade_time)

    def show_portfolio(self):
        self.asset.show_portfolio()

    def trade(self, ticker, buy, trade_time, virtual):
        """
        가격 예측 및 거래 실행
        : current_price: 현재 가격
        : target_return: 목표 수익률
        """
        if virtual:
            idx = 'close' if trade_time.second > 30 else 'open'
            current_price = pyupbit.get_ohlcv(ticker, interval='minute1', count=1, to=trade_time)[idx].iloc[0]
        else:
            current_price = pyupbit.get_current_price(ticker=ticker)
        if buy:
            # 예상 수익률 달성을 위한 구매 수량 계산
            quantity_to_buy = self.asset.cash // current_price
            print(f"quantity_to_buy: {quantity_to_buy}")
            if quantity_to_buy > 0:
                self.buy_crypto(ticker, quantity_to_buy, current_price, trade_time)
                print(f"Bought {quantity_to_buy} units of {ticker} at {current_price} each.")

        # 예측 가격이 현재 가격보다 낮을 것으로 예상되면 판매
        else:
            if ticker in self.asset.portfolio and self.asset.portfolio[ticker][0] > 0:
                quantity_to_sell = self.asset.portfolio[ticker][0]
                self.sell_crypto(ticker, quantity_to_sell, current_price, trade_time)
                print(f"Sold {quantity_to_sell} units of {ticker} at {current_price} each.")


"""
Virtual Trading
예측 값은 close 가격
1. 500 개의 데이터 셋으로 다음날 가격차(=종가 - 시가)를 예측
2. a일 오전 8시 30분에 예측 시작
   - 과거 500개의 15분 봉으로 예측
   - 매매는 오전 9시에 진행
3. 다음날 가격차가 목표 수익률보다 높을경우 현금의 20% 매수 진행
4. 다음날 가격차가 목표 수익률보다 낮을경우 매수하지 않음

###가정###
1. 

"""
#데이터 전처리 함수
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

#데이터셋 생성 함수
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0:5])
        Y.append([data[i + time_step, 0], data[i + time_step, 3]])  # Open과 Close 가격
    return np.array(X), np.array(Y)

def main():
    ticker = "KRW-BTC"
    interval = "minute15"
    name = "VirtualTradier"
    test_days = 15 # 365일간 예측

    user = User(name)

    #현재 날짜를 기준으로 test_days 까지의 ohlcv 수집
    day_ago = datetime.now() - timedelta(days=test_days)

    #가상 시간 설정
    virtual_time_now = day_ago
    virtual_time_before = day_ago

    df = fetch_data_virtual_trading(ticker=ticker, interval=interval, start_date=day_ago)
    print(len(df))
    scaled_data, scaler = preprocess_data(df)

    #500개 데이터로 예측
    time_step = 500

    i = 0
    start_time = time.time()
    end_time = time.time()
    trade = 0
    #while virtual_time_now < datetime.now():
    while True:
        #if virtual_time_now.hour > 8 and virtual_time_now.minute > 30 and trade == 0:
        if True:
            trade = 1
            virtual_time_now += timedelta(end_time - start_time)
            end_time = time.time()
            dataset = scaled_data[i:i+time_step, :]

            #데이터셋 준비
            X, Y = create_dataset(dataset, time_step)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # StockPriceLSTM 모델 생성 및 훈련
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = StockPriceLSTM(input_shape=input_shape, lstm_units=50, stack_depth=2, dropout_rate=0.2)
            model.compile(learning_rate=0.001)
            history = model.train(X_train, y_train, epochs=10, batch_size=32)
            save_history(history, i, name)

            predictions = model.model.predict(X_test)
            print(f"predictions: {predictions}")
            predictions = scaler.inverse_transform(predictions)  # 예측값을 원래 스케일로 변환
            user.trade(ticker=ticker, predict=predictions, trade_time=virtual_time_now)
            virtual_time_now = virtual_time_before
            start_time = time.time()

        elif virtual_time_now.hour > 9 and trade == 1:
            trade = 0

if __name__ == "main":
    main()