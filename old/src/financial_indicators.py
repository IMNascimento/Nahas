import pandas as pd
import numpy as np
from src.utils.technical_indicators import TechnicalIndicators  # Importando a classe de indicadores técnicos

class FinancialIndicators:
    """
    Classe responsável por calcular indicadores técnicos em um DataFrame de preços.
    O DataFrame deve conter pelo menos uma coluna chamada 'Close' com os preços de fechamento.
    Indicadores como SMA, EMA, Envelopes e outros serão adicionados ao DataFrame como novas colunas.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa a classe com um DataFrame de dados.
        :param df: DataFrame contendo pelo menos uma coluna 'Close' com os preços de fechamento.
        """
        if 'Close' not in df.columns:
            raise ValueError("O DataFrame deve conter uma coluna 'close' com os preços de fechamento.")
        self.df = df

    def calcular_sma(self, period: int = 14):
        """
        Calcula a Média Móvel Simples (SMA) para um determinado período.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        """
        self.df[f'SMA_{period}'] = TechnicalIndicators.sma(self.df['Close'], period)
        # Preencher NaN com a média da SMA
        self.df[f'SMA_{period}'].fillna(method='ffill', inplace=True)

    def calcular_ema(self, period: int = 14):
        """
        Calcula a Média Móvel Exponencial (EMA) para um determinado período.
        :param period: Período para o cálculo da EMA. Padrão é 14.
        """
        self.df[f'EMA_{period}'] = TechnicalIndicators.ema(self.df['Close'], period)
        # Preencher NaN com a média da EMA
        self.df[f'EMA_{period}'].fillna(method='ffill', inplace=True)

    def calcular_envelopes(self, period: int = 14, percent: float = 3.0):
        """
        Calcula Envelopes de preço ao redor da SMA para um determinado período e percentual.
        :param period: Período para o cálculo da SMA base para os Envelopes. Padrão é 14.
        :param percent: Percentual para calcular os envelopes superior e inferior. Padrão é 3%.
        """
        envelopes = TechnicalIndicators.envelopes(self.df['Close'], period, percent)
        self.df = pd.concat([self.df, envelopes], axis=1)
        # Preencher NaN para envelopes superior e inferior
        self.df.fillna(method='ffill', inplace=True)

    def calcular_rsi(self, period: int = 14):
        """
        Calcula o Índice de Força Relativa (RSI) para um determinado período.
        :param period: Período para o cálculo do RSI. Padrão é 14.
        """
        self.df[f'RSI_{period}'] = TechnicalIndicators.rsi(self.df['Close'], period)
        # Preencher NaN com forward fill para evitar distorções
        self.df[f'RSI_{period}'].fillna(method='ffill', inplace=True)

    def calcular_macd(self):
        """
        Calcula o MACD (Moving Average Convergence Divergence) e adiciona as colunas MACD, MACD_Signal e MACD_Hist ao DataFrame.
        """
        macd_data = TechnicalIndicators.macd(self.df['Close'])
        self.df = pd.concat([self.df, macd_data], axis=1)
        # Preencher NaN com forward fill
        self.df.fillna(method='ffill', inplace=True)

    def calcular_bollinger_bands(self, period: int = 20, std_dev: float = 2.0):
        """
        Calcula as Bandas de Bollinger para um determinado período e desvio padrão.
        :param period: Período para o cálculo da SMA. Padrão é 20.
        :param std_dev: Desvio padrão para as bandas. Padrão é 2.0.
        """
        self.df['Bollinger_Middle'] = TechnicalIndicators.sma(self.df['Close'], period)
        rolling_std = self.df['Close'].rolling(window=period).std()
        self.df['Bollinger_Upper'] = self.df['Bollinger_Middle'] + std_dev * rolling_std
        self.df['Bollinger_Lower'] = self.df['Bollinger_Middle'] - std_dev * rolling_std
        # Preencher NaN com forward fill
        self.df.fillna(method='ffill', inplace=True)

    def calcular_estocastico(self, period: int = 14):
        """
        Calcula o Oscilador Estocástico.
        :param period: Período para o cálculo do estocástico. Padrão é 14.
        """
        low_min = self.df['Low'].rolling(window=period).min()
        high_max = self.df['High'].rolling(window=period).max()

        # Evitar divisão por zero
        denominator = (high_max - low_min)
        denominator[denominator == 0] = np.nan  # Tratamento para evitar divisões por zero

        self.df['Stochastic_K'] = 100 * (self.df['Close'] - low_min) / denominator
        self.df['Stochastic_D'] = self.df['Stochastic_K'].rolling(window=3).mean()
        # Preencher NaN com forward fill
        self.df.fillna(method='ffill', inplace=True)

    def calcular_atr(self, period: int = 14):
        """
        Calcula o Average True Range (ATR) para medir a volatilidade.
        :param period: Período para o cálculo do ATR. Padrão é 14.
        """
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift()).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['ATR'] = true_range.rolling(window=period).mean()
        # Preencher NaN com forward fill
        self.df.fillna(method='ffill', inplace=True)

    def calcular_todos_indicadores(self):
        """
        Calcula todos os indicadores de uma vez e adiciona-os ao DataFrame.
        Inclui SMA, EMA, Envelopes, RSI e MACD.
        """
        self.calcular_sma(period=14)
        self.calcular_sma(period=20)
        self.calcular_ema(period=7)
        self.calcular_ema(period=14)
        self.calcular_ema(period=20)
        self.calcular_envelopes(period=14, percent=3.0)
        self.calcular_envelopes(period=14, percent=5.0)
        self.calcular_rsi(period=14)
        self.calcular_bollinger_bands(period=20)
        self.calcular_estocastico(period=14)
        self.calcular_atr(period=14)
        self.calcular_macd()

    def get_dataframe(self):
        """
        Retorna o DataFrame com todos os indicadores calculados.
        """
        return self.df