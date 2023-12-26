from utils import *

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

    def save(self, portfolio_filename, transactions_filename):
        # 포트폴리오를 CSV 파일로 저장
        with open(portfolio_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Ticker', 'Quantity', 'Price'])
            for ticker, (quantity, price) in self.portfolio.items():
                writer.writerow([ticker, quantity, price])

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
        self.asset = Asset()  # 사용자의 자산 관리를 위한 Asset 인스턴스

    def add_cash(self, amount):
        self.asset.add_cash(amount)

    def buy_crypto(self, ticker, quantity, price):
        return self.asset.buy(ticker, quantity, price)

    def sell_crypto(self, ticker, quantity, price):
        return self.asset.sell(ticker, quantity, price)

    def show_portfolio(self):
        self.asset.show_portfolio()

# 자산 클래스의 인스턴스 생성 및 테스트
my_asset = Asset()
my_asset.add_cash(1000)  # 현금 추가
my_asset.buy('BTC', 2, 200)  # BTC 구매
my_asset.sell('BTC', 1, 250)  # BTC 판매

# 포트폴리오 및 거래 기록 저장
portfolio_filename = 'data/my_portfolio.csv'
transactions_filename = 'data/my_transactions.csv'
my_asset.save(portfolio_filename, transactions_filename)

portfolio_filename, transactions_filename  # 파일 경로 반환

