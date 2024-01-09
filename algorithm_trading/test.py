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