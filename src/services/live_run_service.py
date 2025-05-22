import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from config.settings import Settings
from services.binance import BinanceData

import importlib

def get_trainer_class(framework: str, model_type: str):
    """
    Importa e retorna a classe trainer correta conforme o framework e model_type.
    """
    module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"
    class_name = f"{framework.capitalize()}{model_type.capitalize()}Trainer"
    try:
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Trainer não encontrado: {module_path}.{class_name}\nErro: {e}")

def get_latest_binance_data(window_hours: int, symbol: str, interval: str) -> pd.DataFrame:
    binance = BinanceData()
    agora = datetime.utcnow()
    if agora.minute != 0 or agora.second != 0 or agora.microsecond != 0:
        agora -= timedelta(hours=1)
    delta = timedelta(minutes=agora.minute, seconds=agora.second, microseconds=agora.microsecond)
    end_time = agora - delta
    start_time = end_time - timedelta(hours=window_hours)
    start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
    end_str = end_time.strftime("%d %b, %Y %H:%M:%S")
    df = binance.get_historical_data(symbol, start_str=start_str, interval=interval, end_str=end_str)
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def prepare_data_for_model(data: pd.DataFrame, processor: DataProcessor, relevant_columns, window_size) -> np.ndarray:
    features = data[relevant_columns].values
    features_normalized = processor.normalize_sliding_window(features)
    window_normalized = features_normalized[-window_size:]
    X_input = np.array(window_normalized, dtype=np.float32).reshape(1, window_normalized.shape[0], window_normalized.shape[1])
    return X_input

def live_run(
    symbol: str,
    interval: str,
    window_hours: int,
    framework: str,
    model_type: str,
    model_path: str,
    window_size: int,
    relevant_columns: list,
    processor_kwargs: dict = {},
    predict_kwargs: dict = {},
    **trainer_kwargs
):
    """
    Execução online genérica para qualquer modelo.
    """
    # Importa o trainer correto
    trainer_class = get_trainer_class(framework, model_type)
    trainer = trainer_class(**trainer_kwargs)
    processor = DataProcessor(window_size=window_size, **processor_kwargs)

    # Carrega o modelo salvo
    if hasattr(trainer, "load_model"):
        model = trainer.load_model(model_path)
    elif hasattr(trainer, "loading_model"):
        model = trainer.loading_model(model_path)
    else:
        raise RuntimeError("Trainer não tem método de carregar modelo salvo.")

    print(f"Iniciando Live Run com {framework}/{model_type} ({model_path})...")
    while True:
        try:
            data = get_latest_binance_data(window_hours, symbol, interval)
            if len(data) < window_size:
                print("Dados insuficientes para previsão. Aguardando novos dados...")
                time.sleep(60)
                continue

            X_input = prepare_data_for_model(data, processor, relevant_columns, window_size)

            # Predição universal
            if hasattr(model, "predict"):
                predicted_scaled = model.predict(X_input, **predict_kwargs)
            elif hasattr(trainer, "predict"):
                predicted_scaled = trainer.predict(model, X_input)
            else:
                raise RuntimeError("Trainer/modelo não possui método predict.")

            # Desnormalização (ajuste se precisar para seu modelo!)
            features = data[relevant_columns].values
            mean = features[-window_size:].mean(axis=0)
            std = features[-window_size:].std(axis=0) + 1e-8
            predicted_value = (predicted_scaled * std[-1]) + mean[-1]

            print(f"Previsão para {symbol}: {predicted_value[0][0]:.2f} (Preço atual: {data.iloc[-1]['close']:.2f})")

            # Espera até o próximo candle
            now = datetime.utcnow()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            sleep_time = (next_hour - now).total_seconds()
            print(f"Aguardando {round(sleep_time/60, 2)} minutos até a próxima previsão.")
            time.sleep(max(sleep_time, 60))
        except Exception as e:
            import traceback
            print(f"Erro no live_run: {e}")
            traceback.print_exc()
            time.sleep(10)
