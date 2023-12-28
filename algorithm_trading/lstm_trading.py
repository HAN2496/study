from utils import *
#from utils import StockPriceLSTM

class Asset:
    def __init__(self):
        self.portfolio = {}  # 형식: {ticker: [수량, 가격]}
        self.cash = 0
        self.total_asset = 0
        self.transaction_history = []  # 거래 기록

    def add_cash(self, amount):
        self.cash += amount
        self.update_total_asset()

    def buy(self, ticker, quantity, price):
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
        self.record_transaction('buy', ticker, quantity, price)
        return True

    def sell(self, ticker, quantity, price):
        if ticker not in self.portfolio or self.portfolio[ticker][0] < quantity:
            print("Insufficient asset to sell.")
            return False
        self.portfolio[ticker][0] -= quantity
        self.cash += quantity * price
        if self.portfolio[ticker][0] == 0:
            del self.portfolio[ticker]
        self.update_total_asset()
        self.record_transaction('sell', ticker, quantity, price)
        return True

    def update_total_asset(self):
        self.total_asset = self.cash + sum(quantity * price for quantity, price in self.portfolio.values())

    def record_transaction(self, transaction_type, ticker, quantity, price):
        self.transaction_history.append({
            "type": transaction_type,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now()
        })

    def show_portfolio(self):
        print("Current Portfolio:")
        for ticker, (quantity, price) in self.portfolio.items():
            print(f"{ticker}: {quantity} units at {price} each")
        print(f"Total Asset: {self.total_asset}")
        print(f"Cash: {self.cash}")
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

    def add_cash(self, amount):
        self.asset.add_cash(amount)

    def buy_crypto(self, ticker, quantity, price):
        return self.asset.buy(ticker, quantity, price)

    def sell_crypto(self, ticker, quantity, price):
        return self.asset.sell(ticker, quantity, price)

    def show_portfolio(self):
        self.asset.show_portfolio()

    def predict_and_trade(self, ticker, X_test, current_price, target_return):
        """
        가격 예측 및 거래 실행
        :param ticker: 거래할 티커
        :param X_test: 예측에 사용될 데이터
        :param current_price: 현재 가격
        :param target_return: 목표 수익률
        """
        predicted_price = self.lstm_model.model.predict(X_test)[0][0]

        # 예측 가격이 현재 가격 대비 목표 수익률 이상 증가할 것으로 예상되면 구매
        if predicted_price >= current_price * (1 + target_return):
            # 예상 수익률 달성을 위한 구매 수량 계산 (단순 예시)
            quantity_to_buy = self.asset.cash // current_price
            if quantity_to_buy > 0:
                self.buy_crypto(ticker, quantity_to_buy, current_price)
                print(f"Bought {quantity_to_buy} units of {ticker} at {current_price} each.")

        # 예측 가격이 현재 가격보다 낮을 것으로 예상되면 판매
        elif predicted_price < current_price:
            if ticker in self.asset.portfolio and self.asset.portfolio[ticker][0] > 0:
                quantity_to_sell = self.asset.portfolio[ticker][0]
                self.sell_crypto(ticker, quantity_to_sell, current_price)
                print(f"Sold {quantity_to_sell} units of {ticker} at {current_price} each.")

class VirtualTrading:
    def __init__(self, initial_budget=10000000):
        self.initial_budget = initial_budget
        self.user_name = "Virtual Trader"
        self.user = User(self.user_name)
        self.user.add_cash(initial_budget)
        self.trading_history = []
        self.lstm_model = StockPriceLSTM(input_shape=lstm_input_shape)

    def prepare_data(self, data, window_size=60):
        """
        LSTM 모델 입력을 위한 데이터 준비
        :param data: 원시 시장 데이터
        :param window_size: 사용할 시퀀스의 길이
        :return: 변환된 데이터
        """
        features = data[['close', 'high', 'low', 'volume']].values

        # 데이터 정규화
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)

        # 시퀀스 데이터 생성
        X_test = []
        for i in range(window_size, len(scaled_data)):
            X_test.append(scaled_data[i - window_size:i, :])

        return np.array(X_test)


    def fetch_data(self, ticker, interval="minute15", count=300):
        """
        PyUpbit로부터 특정 티커의 과거 데이터를 불러옵니다.
        :param ticker: 데이터를 불러올 티커
        :param interval: 데이터 간격 (예: 15분봉)
        :param count: 불러올 데이터의 수
        :return: 과거 시장 데이터
        """
        return pyupbit.get_ohlcv(ticker, interval=interval, count=count)

    def simulate_trading(self, ticker, target_return, lstm_model, window_size=60):
        """
        가상 거래 시뮬레이션
        :param ticker: 거래할 티커
        :param target_return: 목표 수익률
        :param lstm_model: LSTM 모델 인스턴스
        :param window_size: 사용할 시퀀스의 길이
        """
        historical_data = self.fetch_data(ticker)

        for date, data in historical_data.iterrows():
            # 시뮬레이션을 위한 데이터 준비
            X_test = self.prepare_data(historical_data[:date], window_size=window_size)
            current_price = data['close']

            # LSTM 모델을 사용한 예측
            predicted_price = lstm_model.model.predict(X_test[-1].reshape(1, window_size, -1))[0][0]

            # 예측 및 거래 실행
            self.user.predict_and_trade(ticker, predicted_price, current_price, target_return)

            # 거래 후 포트폴리오 상태 및 거래 기록을 저장
            self.trading_history.append({
                "date": date,
                "portfolio": self.user.asset.portfolio,
                "cash": self.user.asset.cash,
                "total_asset": self.user.asset.total_asset
            })

        # 거래 결과 저장
        transactions_filename = "virtual_trader_transactions.csv"
        self.user.asset.save(transactions_filename)

    def get_trading_history(self):
        """
        거래 기록 반환
        """
        return self.trading_history


if __name__ == "__main__":
    virtualTrader = VirtualTrading()
    virtualTrader.simulate_trading("KRW-BTC", 0.01)
