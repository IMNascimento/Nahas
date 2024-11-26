from models.db.model_binance import HourlyQuote  # Modelo do banco
from data.data_processing import DataProcessor  # Sua classe DataProcessor
from models.lstm_model import CustomLSTMTrainer  # Sua classe CustomLSTMTrainer
import pandas as pd
import numpy as np
import os

# Configurações
window_size = 48  # Tamanho da janela de entrada
steps_ahead = 1  # Previsão de 1 passo à frente
scaler_path = "result/scaler"  # Caminho para salvar os scalers

# Recuperar dados do banco
def load_data_from_db():
    """
    Carrega os dados do banco de dados em um DataFrame.
    """
    data = HourlyQuote.select().dicts()  # Seleciona os dados do banco
    data_df = pd.DataFrame(list(data))  # Converte para DataFrame
    data_df.sort_values(by="timestamp", inplace=True)  # Ordena por timestamp
    return data_df

# Carregar dados do banco
data_df = load_data_from_db()
if 'timestamp' in data_df.columns:
    data_df['timestamp'] = data_df['timestamp'].apply(lambda x: x.timestamp())
# Configurar a coluna alvo
coluna_alvo = "close"  # Prever o preço de fechamento

# Inicializar o DataProcessor
processor = DataProcessor(window_size=window_size)

# Criar janelas de entrada (X) e saída (y)
X, y = processor.create_windows(data=data_df, coluna_alvo=coluna_alvo, steps_ahead=steps_ahead)

# Dividir os dados em treino, validação e teste
X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)

# Normalizar os dados
X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

# Salvar os scalers
processor.save_scaler(path=os.path.join(scaler_path, "scaler.pkl"))

# Definir o shape de entrada para o LSTM
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])  # (timesteps, features)

# Inicializar o treinador LSTM
trainer = CustomLSTMTrainer(
    input_shape=input_shape,
    layers_config=[64, 32],  # Duas camadas LSTM: 64 unidades na 1ª e 32 na 2ª
    dropout=0.2,
    batch_size=16,
    epochs=50,
    patience=5
)

# Treinar o modelo
lstm_model = trainer.train(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)

# Salvar o modelo treinado
trainer.save_model(lstm_model, model_path="result/models/lstm_model_window_48.h5")

# Fazer previsões com o conjunto de teste
y_pred_scaled = lstm_model.predict(X_test_scaled)

# Carregar os scalers para reverter a normalização
processor.load_scaler(path=os.path.join(scaler_path, "scaler.pkl"))

# Reverter a normalização para y_pred e y_test
y_pred = processor.inverse_transform(y_pred_scaled)
y_test = processor.inverse_transform(y_test_scaled)

# Exibir resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Real")
plt.plot(y_pred, label="Previsto")
plt.legend()
plt.title("Previsão de Preço com LSTM")
plt.show()
