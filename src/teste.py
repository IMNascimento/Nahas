from models.db.model_binance import HourlyQuote
from keras.models import load_model
from data.data_processing import DataProcessor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configurações
window_size = 48  # Tamanho da janela
start_date = "2024-11-14 00:00:00"  # Data inicial para buscar os dados
scaler_path = "result/scaler/"
model_path = "result/models/lstm_model_window_48.h5"

# Função para carregar os dados do banco
def load_data_from_db(start_date):
    """
    Carrega os dados a partir de uma data específica do banco de dados.
    """
    query = (HourlyQuote
             .select()
             .where(HourlyQuote.timestamp >= start_date)
             .order_by(HourlyQuote.timestamp))
    
    data = pd.DataFrame(list(query.dicts()))
    if not data.empty:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data[['id','timestamp', 'open', 'high', 'low', 'close', 'volume']]
    else:
        print("Nenhum dado encontrado no banco.")
        return None


# Função para testar o modelo
def test_model(start_date):
    """
    Testa o modelo treinado com os dados do banco e armazena as previsões.
    """
    # Carregar o modelo e scaler
    model = load_model(model_path)
    processor = DataProcessor(window_size=window_size)
    processor.load_scaler(scaler_path)

    # Carregar os dados do banco
    data = load_data_from_db(start_date)
    if data is None or len(data) < window_size:
        print("Dados insuficientes para realizar a previsão.")
        return

    # Carregar dados do banco
    if 'timestamp' in data.columns:
        data['timestamp'] = data['timestamp'].apply(lambda x: x.timestamp())

    # Preparar o array para armazenar os resultados
    results = []

    # Iterar pelas janelas de dados para fazer as previsões
    for i in range(len(data) - window_size):
        # Criar uma janela de 48 horas
        window = data.iloc[i:i + window_size]
        real_value = data.iloc[i + window_size]['close']  # Valor real para a próxima hora

        # Normalizar e preparar os dados para o modelo
        features = window[['id','timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        features_scaled = processor.scaler_X.transform(features)
        X_input = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])

        # Fazer a previsão
        predicted_scaled = model.predict(X_input)
        # Verificar a forma do array e desnormalizar corretamente
        if predicted_scaled.ndim == 1:  # Caso seja 1D
            predicted_value = processor.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
        else:  # Caso seja 2D
            predicted_value = processor.inverse_transform(predicted_scaled)[0, 0]

        # Armazenar o valor real e o previsto
        results.append({
            'timestamp': data.iloc[i + window_size]['timestamp'],
            'real_value': real_value,
            'predicted_value': predicted_value
        })

    # Retornar os resultados como um DataFrame
    return pd.DataFrame(results)

# Executar o teste
results_df = test_model(start_date)

# Exibir os resultados
if results_df is not None:
    print(results_df.head())  # Exibir os primeiros resultados
    print(results_df.tail())  # Exibir os últimos resultados

    # Opcional: Salvar os resultados em CSV
    results_df.to_csv("test_results.csv", index=False)
    print("Resultados salvos em test_results.csv")
