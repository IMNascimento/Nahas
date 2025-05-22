from services.binance import BinanceData


binance = BinanceData()
a = binance.get_historical_data("BTCBRL", start_str="20 Jan, 2023 14:00:00", interval="1h")
print(a)