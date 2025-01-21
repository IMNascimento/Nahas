import os
import sys
import traceback
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from models.lstm_model import CustomLSTMTrainer
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from services.binance import BinanceData
from config.settings import Settings
from models.db.model_binance import HourlyQuote


def get_latest_binance_data(window_hours: int, symbol: str, interval: str) -> pd.DataFrame:
    """
    Obtém os preços de fechamento das últimas 'window_hours' horas para o par especificado.

    :param window_hours: Quantidade de horas para obter dados históricos.
    :param symbol: Par de criptomoeda, ex: 'BTCUSDT'.
    :param interval: Intervalo das velas, ex: '1h'.
    :return: DataFrame contendo os dados históricos.
    """
    binance = BinanceData()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=window_hours)

    # Formatar as strings de datas
    start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
    end_str = end_time.strftime("%d %b, %Y %H:%M:%S")

    # Buscar dados históricos
    df = binance.get_historical_data(symbol, start_str=start_str, interval=interval, end_str=end_str)
    
    if df.empty:
        raise ValueError("Nenhum dado retornado pela Binance. Verifique os parâmetros.")

    # Garantir que as colunas numéricas estão no formato correto
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df


def prepare_data_for_model(data: pd.DataFrame, processor: DataProcessor) -> np.ndarray:
    """
    Prepara os dados para entrada no modelo LSTM.

    :param data: DataFrame com os dados históricos.
    :param processor: Instância de DataProcessor para processar os dados.
    :return: Dados preparados (X_input) para o modelo.
    """
    # Selecionar apenas as colunas relevantes
    features = data[Settings.RELEVANT_COLUMNS].values
    print(features)
    # Normalizar os dados com janela deslizante
    features_normalized = processor.normalize_sliding_window(features)

    # Criar a última janela deslizante para prever a próxima hora
    window_normalized = features_normalized[-Settings.WINDOW_SIZE:]
    X_input = np.array(window_normalized, dtype=np.float32).reshape(1, window_normalized.shape[0], window_normalized.shape[1])

    return X_input


def live_run(symbol, interval, window_hours):
    """
    Executa o modelo em tempo real com dados coletados da Binance.

    :param symbol: Par de negociação (ex: 'BTCUSDT').
    :param interval: Intervalo das velas (ex: '1h').
    :param window_hours: Quantidade de horas para buscar dados históricos.
    """
    trainer = CustomLSTMTrainer()
    model = trainer.loading_model(Settings.LOAD_MODEL_FINETUNING)
    processor = DataProcessor(window_size=Settings.WINDOW_SIZE)

    print("Inicializando Live Run...")
    while True:
        try:
            # Obter dados históricos
            data = get_latest_binance_data(window_hours, symbol, interval)
            print(data.tail())
            print(data)  # Para verificar os dados obtidos

            # Verificar se há dados suficientes para o modelo
            if len(data) < Settings.WINDOW_SIZE:
                print("Dados insuficientes para previsão. Aguardando novos dados...")
                time.sleep(60)
                continue

            # Preparar os dados para o modelo
            X_input = prepare_data_for_model(data, processor)

            # Fazer previsão
            predicted_scaled = model.predict(X_input)

            # Reverter a normalização
            features = data[Settings.RELEVANT_COLUMNS].values
            mean = features[-Settings.WINDOW_SIZE:].mean(axis=0)
            std = features[-Settings.WINDOW_SIZE:].std(axis=0) + 1e-8
            predicted_value = (predicted_scaled * std[-1]) + mean[-1]

            # Exibir resultados
            print(f"Previsão para {symbol}: {predicted_value[0][0]:.2f} (Preço atual: {data.iloc[-1]['close']:.2f})")

            # Calcular tempo até o próximo candle
            current_time = datetime.utcnow()
            sleep_time = 3600 - (current_time.minute * 60 + current_time.second)  # Ajuste para '1h'
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Erro durante a execução do Live Run: {e}")
            traceback.print_exc()
            time.sleep(10)


# Executar
SYMBOL = "BTCUSDT"
INTERVAL = "1h"  # Intervalo de 1 hora

live_run(SYMBOL, INTERVAL, Settings.WINDOW_SIZE)
