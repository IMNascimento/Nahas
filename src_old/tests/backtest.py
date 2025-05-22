import pandas as pd

class Backtest:
    """
    Classe para realizar backtests de estratégias de negociação usando dados históricos.
    """

    def __init__(self, historical_data: pd.DataFrame, initial_balance: float, trading_fee: float = 0.001):
        """
        Inicializa o backtest.

        :param historical_data: DataFrame contendo os dados históricos (deve incluir 'timestamp', 'open', 'high', 'low', 'close').
        :param initial_balance: Saldo inicial em moeda base (ex: USDT).
        :param trading_fee: Taxa de negociação em percentual (ex: 0.001 para 0.1%).
        """
        self.data = historical_data
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.balance = initial_balance
        self.position = 0.0  # Quantidade da criptomoeda em carteira
        self.trades = []  # Histórico de trades
        self.total_profit = 0.0

    def simulate(self, strategy):
        """
        Realiza o backtest aplicando a estratégia fornecida.

        :param strategy: Função que recebe um DataFrame e retorna um sinal ('BUY', 'SELL', ou 'HOLD').
        """
        for index, row in self.data.iterrows():
            # Recebe o sinal da estratégia
            signal = strategy(row)

            if signal == 'BUY' and self.balance > 0:
                self._buy(row['close'], row['timestamp'])
            elif signal == 'SELL' and self.position > 0:
                self._sell(row['close'], row['timestamp'])

        # Fecha qualquer posição aberta no final do backtest
        if self.position > 0:
            self._sell(self.data.iloc[-1]['close'], self.data.iloc[-1]['timestamp'])

    def _buy(self, price, timestamp):
        """
        Executa uma compra no backtest.
        """
        # Cálculo da quantidade comprada (descontando a taxa de negociação)
        qty = (self.balance * (1 - self.trading_fee)) / price
        self.position += qty
        self.trades.append({'timestamp': timestamp, 'type': 'BUY', 'price': price, 'quantity': qty})
        self.balance = 0.0

    def _sell(self, price, timestamp):
        """
        Executa uma venda no backtest.
        """
        # Cálculo do valor da venda (descontando a taxa de negociação)
        sell_value = self.position * price * (1 - self.trading_fee)
        profit = sell_value - self.initial_balance  # Lucro em USDT
        self.total_profit += profit
        self.balance += sell_value
        self.trades.append({'timestamp': timestamp, 'type': 'SELL', 'price': price, 'quantity': self.position})
        self.position = 0.0

    def get_results(self):
        """
        Retorna os resultados do backtest.
        """
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_profit': self.total_profit,
            'trade_count': len(self.trades),
            'trades': pd.DataFrame(self.trades)
        }
    



# Suponha que você já tem os dados históricos no DataFrame `historical_data`
#initial_balance = 1000  # Saldo inicial em USDT
#trading_fee = 0.001  # 0.1%

# Inicializa o backtest
#backtest = Backtest(historical_data, initial_balance, trading_fee)

# Simula a estratégia
#backtest.simulate(simple_moving_average_strategy)

# Obtém os resultados
#results = backtest.get_results()

# Exibe os resultados
#print("Saldo inicial:", results['initial_balance'])
#print("Saldo final:", results['final_balance'])
#print("Lucro total:", results['total_profit'])
#print("Número de trades:", results['trade_count'])

# Visualiza o histórico de trades
#print(results['trades'])

#def simple_moving_average_strategy(row):
    """
    Estratégia simples baseada no preço de fechamento.
    Compra quando o preço de fechamento é menor que 50.000 e vende quando é maior que 60.000.
    """
#    if row['close'] < 50000:
#        return 'BUY'
#    elif row['close'] > 60000:
#        return 'SELL'
#    return 'HOLD'