from binance.client import Client
from config.settings import Settings
from datetime import datetime
import pandas as pd

class BinanceData:
    """
    Classe responsável pela conexão com a API da Binance e coleta de dados históricos de criptomoedas.
    """

    def __init__(self):
        """
        Inicializa a classe carregando as credenciais da API da Binance do arquivo .env.
        """
        # Inicializa o cliente da Binance com as credenciais
        self.client = Client(Settings.BINANCE_API_KEY, Settings.BINANCE_API_SECRET_KEY)

    def get_historical_data(self, symbol: str, start_str: str, interval: str, end_str: str = None) -> pd.DataFrame:
        """
        Obtém dados históricos da criptomoeda em um intervalo de tempo específico.

        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :param start_str: Data de início (em string), ex: '1 Jan, 2020'.
        :param interval: Intervalo de tempo, ex: Client.KLINE_INTERVAL_1HOUR para 1 hora.
        :param end_str: Data final (opcional), ex: '1 Jan, 2023'.
        :return: Um DataFrame contendo os dados históricos.
        """
        # Coleta os dados históricos de candles (klines)
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)

        # Criar um DataFrame a partir dos dados coletados
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Converter o timestamp para um formato de data legível
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Retornar apenas as colunas mais importantes
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_live_data(self, symbol: str) -> dict:
        """
        Obtém dados de preço ao vivo para um par de criptomoedas.
        
        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :return: Dicionário contendo os dados de preço ao vivo.
        """
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return {
            'symbol': ticker['symbol'],
            'price': float(ticker['price']),
            'time': datetime.now()
        }

    def place_order(self, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        """
        Coloca uma ordem de compra ou venda na Binance.
        
        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :param side: 'BUY' para compra ou 'SELL' para venda.
        :param quantity: Quantidade da criptomoeda a comprar/vender.
        :param price: Preço limite (opcional, usado para ordens limitadas).
        :return: Detalhes da ordem realizada.
        """
        try:
            if price:
                # Ordem limitada
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=quantity,
                    price=f"{price:.8f}"
                )
            else:
                # Ordem de mercado
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
            return order
        except Exception as e:
            print(f"Erro ao colocar a ordem: {e}")
            return {'error': str(e)}

    def check_balance(self, asset: str) -> float:
        """
        Verifica o saldo disponível para um ativo.
        
        :param asset: Código do ativo, ex: 'BTC', 'USDT'.
        :return: Quantidade disponível do ativo.
        """
        account_info = self.client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancela uma ordem aberta.
        
        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :param order_id: ID da ordem a ser cancelada.
        :return: Detalhes da ordem cancelada.
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            return result
        except Exception as e:
            print(f"Erro ao cancelar a ordem: {e}")
            return {'error': str(e)}

    def get_open_orders(self, symbol: str = None) -> list:
        """
        Obtém todas as ordens abertas para um par de criptomoedas específico ou para todas.
        
        :param symbol: Par de criptomoedas (opcional), ex: 'BTCUSDT'.
        :return: Lista de ordens abertas.
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            return orders
        except Exception as e:
            print(f"Erro ao obter ordens abertas: {e}")
            return {'error': str(e)}

    def get_trade_history(self, symbol: str) -> pd.DataFrame:
        """
        Obtém o histórico de trades realizados para um par de criptomoedas.
        
        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :return: DataFrame com o histórico de trades.
        """
        try:
            trades = self.client.get_my_trades(symbol=symbol)
            df = pd.DataFrame(trades)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df
        except Exception as e:
            print(f"Erro ao obter histórico de trades: {e}")
            return pd.DataFrame({'error': [str(e)]})
        



# Inicializando a classe
#binance_data = BinanceData()

# Verificando saldo de USDT
#balance = binance_data.check_balance('USDT')
#print(f"Saldo disponível de USDT: {balance}")

# Comprando 0.001 BTC
#order = binance_data.place_order('BTCUSDT', 'BUY', quantity=0.001)
#print(f"Detalhes da ordem: {order}")

# Obtendo preço ao vivo
#live_data = binance_data.get_live_data('BTCUSDT')
#print(f"Preço ao vivo do BTC/USDT: {live_data['price']}")