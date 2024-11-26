from keras.models import load_model
import numpy as np
import pandas as pd
from data.data_processing import DataProcessor
from models.db.model_binance import HourlyQuote

# Caminhos dos arquivos
model_path = "result/models/lstm_model_window_48.h5"
scaler_path = "result/scaler"

# Configurações
window_size = 48  # Tamanho da janela
coluna_alvo = "close"  # Prevendo o preço de fechamento

# Carregar o modelo e scalers
def load_model_and_scaler():
    """
    Carrega o modelo e os scalers salvos.
    """
    # Carregar o modelo salvo
    model = load_model(model_path)
    
    # Inicializar o processador e carregar scalers
    processor = DataProcessor(window_size=window_size)
    processor.load_scaler(path=scaler_path)
    
    return model, processor

# Carregar os dados do banco
def get_latest_data():
    """
    Recupera os últimos 48 registros do banco para fazer a previsão.
    """
    data = HourlyQuote.select().order_by(HourlyQuote.timestamp.desc()).limit(window_size).dicts()
    data_df = pd.DataFrame(list(data))
    data_df.sort_values(by="timestamp", inplace=True)  # Ordenar os dados
    return data_df

# Fazer a previsão
def predict_today():
    """
    Faz a previsão para o próximo valor com base nos últimos 48 registros.
    """
    # Carregar modelo e scalers
    model, processor = load_model_and_scaler()

    # Carregar os últimos dados
    data_df = get_latest_data()

    # Garantir que somente as colunas relevantes sejam usadas (sem timestamp)
    features = data_df.drop(columns=["timestamp", "id"]).values  # Ajuste conforme suas colunas

    # Normalizar os dados
    features_scaled = processor.scaler_X.transform(features)

    # Formatar no formato 3D esperado pelo modelo (1, timesteps, features)
    X_input = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])

    # Fazer a previsão
    y_pred_scaled = model.predict(X_input)

    # Reverter a normalização da previsão
    y_pred = processor.inverse_transform(y_pred_scaled)

    return y_pred[0]  # Retornar o valor previsto

# Executar a previsão de hoje
previsao_hoje = predict_today()
print(f"Previsão para o preço de fechamento de hoje: {previsao_hoje}")