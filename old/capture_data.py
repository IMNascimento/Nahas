from src.metatrader_data import MetaTraderData
from src.yahoofinance_data import YahooFinanceData
from src.utils.csv_handler import CSVHandler
from src.binance_data import BinanceData
from binance.client import Client
from datetime import datetime
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os


# Uso das classes MetaTraderData e CSVHandler
if __name__ == "__main__":
    load_dotenv()
  
    mt_data = MetaTraderData(login=int(os.getenv('LOGIN_MT5')), password=os.getenv('PASSWORD_MT5'), server=os.getenv('SERVER_MT5'))
    
    # Inicializa a conexão com o MetaTrader 5
    mt_data.initialize()
    
    empresas = ['ITSA4', 'PETR4', 'VALE3', 'BBDC4', 'ABEV3', 'B3SA3', 'ITUB4', 'PETR3', 'BBAS3', 'BBDC3']

    yahoo_data = YahooFinanceData()
    # Lista de ativos (símbolos no Yahoo Finance)
    ativos = ['BTC-USD', 'ITSA4.SA', 'PETR4.SA','VALE3.SA', 'BBDC4.SA', 'ABEV3.SA', 'B3SA3.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'BBDC3.SA']
    
    binance_data = BinanceData(api_secret=os.getenv('BINANCE_API_SECRET_KEY'), api_key=os.getenv('BINANCE_API_KEY'))
    btc_pairs = ['BTCUSDT', 'BTCBRL']
 
    data = [
        [2024, 1, 1, 2024, 9, 14],
        [2023, 1, 1, 2023, 12, 31],
        [2022, 1, 1, 2022, 12, 31],
        [2021, 1, 1, 2021, 12, 31],
        [2020, 1, 1, 2020, 12, 31],
        [2019, 1, 1, 2019, 12, 31],
        [2018, 1, 1, 2018, 12, 31],
        [2017, 1, 1, 2017, 12, 31],
        [2016, 1, 1, 2016, 12, 31],
        [2015, 1, 1, 2015, 12, 31],
        [2014, 1, 1, 2014, 12, 31],
        [2013, 1, 1, 2013, 12, 31]
    ]

    try:

         # Itera sobre as empresas
        for empresa in empresas:
            # Itera sobre o array de datas
            for start_year, start_month, start_day, end_year, end_month, end_day in data:
                start_date = datetime(start_year, start_month, start_day)
                end_date = datetime(end_year, end_month, end_day)
                
                # Coleta os dados de ticks para o intervalo de datas e empresa atual
                ticks_df = mt_data.get_ticks(empresa, start_date, end_date)
                ticks_filename = f'{empresa}_ticks_{start_year}_{end_year}.csv'
                CSVHandler.save_to_csv(ticks_df, ticks_filename, 'output/metatrader/ticks')
                
                # Coleta os dados de candles para o intervalo de datas e empresa atual
                candles_df = mt_data.get_candles(empresa, mt5.TIMEFRAME_MN1, start_date, end_date)
                candle_filename = f'{empresa}_candles_month_{start_year}_{end_year}.csv'
                CSVHandler.save_to_csv(candles_df, candle_filename, f'output/metatrader/candles/{empresa}/monthly')
        
        for ativo in ativos:
            # Itera sobre o array de datas
            for start_year, start_month, start_day, end_year, end_month, end_day in data:
                start_date = datetime(start_year, start_month, start_day)
                end_date = datetime(end_year, end_month, end_day)

                # Converte as datas para string no formato 'YYYY-MM-DD'
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                try:
                    # Coleta os dados históricos de cada ativo no intervalo de datas
                    asset_data = yahoo_data.get_historical_data(ativo, start_str, end_str, '1mo')

                    # Gera o nome do arquivo CSV e o diretório correto
                    yahoo_filename = f'{ativo}_month_{start_year}_{end_year}.csv'
                    folder = f'output/yahoo/{ativo}/month'  # Corrigido o caminho da pasta

                    # Salva os dados no diretório output/yahoo/ativo
                    CSVHandler.save_to_csv(asset_data, yahoo_filename, folder)

                except Exception as e:
                    print(f"Erro ao coletar ou salvar dados de {ativo}: {e}")

        # Itera sobre os pares de criptomoedas
        for pair in btc_pairs:
            # Itera sobre o array de datas
            for start_year, start_month, start_day, end_year, end_month, end_day in data:
                start_date = datetime(start_year, start_month, start_day)
                end_date = datetime(end_year, end_month, end_day)

                # Converte as datas para string para passar para a API da Binance
                start_str = start_date.strftime("%d %b, %Y")
                end_str = end_date.strftime("%d %b, %Y")

                # Coleta os dados históricos de cada par no intervalo de datas
                btc_data = binance_data.get_historical_data(pair, start_str, Client.KLINE_INTERVAL_1MONTH, end_str)

                # Gera o nome do arquivo CSV
                binance_filename = f'{pair}_month_{start_year}_{end_year}.csv'

                # Salva os dados no diretório output/bitcoin/binance
                CSVHandler.save_to_csv(btc_data, binance_filename, 'output/binance/bitcoin/month')
    
        
    finally:
        # Finaliza a conexão
        mt_data.shutdown()









#yahoo finance
#'1m'	1 minuto
#'2m'	2 minutos
#'5m'	5 minutos
#'15m'	15 minutos
#'30m'	30 minutos
#'60m' / '1h'	1 hora
#'1d'	1 dia (diário)
#'5d'	5 dias
#'1wk'	1 semana
#'1mo'	1 mês
#'3mo'	3 meses


#meta trader
#mt5.TIMEFRAME_M1	1 minuto
#mt5.TIMEFRAME_M5	5 minutos
#mt5.TIMEFRAME_M15	15 minutos
#mt5.TIMEFRAME_M30	30 minutos
#mt5.TIMEFRAME_H1	1 hora
#mt5.TIMEFRAME_H4	4 horas
#mt5.TIMEFRAME_D1	1 dia (diário)
#mt5.TIMEFRAME_W1	1 semana
#mt5.TIMEFRAME_MN1	1 mês





#binance
#Client.KLINE_INTERVAL_1MINUTE	1 minuto
#Client.KLINE_INTERVAL_3MINUTE	3 minutos
#Client.KLINE_INTERVAL_5MINUTE	5 minutos
#Client.KLINE_INTERVAL_15MINUTE	15 minutos
#Client.KLINE_INTERVAL_30MINUTE	30 minutos
#Client.KLINE_INTERVAL_1HOUR	1 hora
#Client.KLINE_INTERVAL_2HOUR	2 horas
#Client.KLINE_INTERVAL_4HOUR	4 horas
#Client.KLINE_INTERVAL_6HOUR	6 horas
#Client.KLINE_INTERVAL_8HOUR	8 horas
#Client.KLINE_INTERVAL_12HOUR	12 horas
#Client.KLINE_INTERVAL_1DAY	1 dia
#Client.KLINE_INTERVAL_3DAY	3 dias
#Client.KLINE_INTERVAL_1WEEK	1 semana
#Client.KLINE_INTERVAL_1MONTH	1 mês