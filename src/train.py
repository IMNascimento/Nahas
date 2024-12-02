from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.lstm_model import CustomLSTMTrainer
from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Configurações
WINDOW_SIZE = 48
STEPS_AHEAD = 1
MODEL_PATH = "result/models/lstm_model_window_48.h5"
END_DATE = "2024-09-14 23:59:59"
RELEVANT_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"
CSV_OUTPUT_PATH = "treino.csv"

def load_data_from_db(end_date):
    """
    Carrega os dados do banco de dados até uma data específica.
    """
    query = (HourlyQuote
             .select()
             .where(HourlyQuote.timestamp <= end_date)
             .order_by(HourlyQuote.timestamp))
    data = pd.DataFrame(list(query.dicts()))
    if not data.empty:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    else:
        print("Nenhum dado encontrado no banco.")
        return pd.DataFrame()

def main():
    data_df = load_data_from_db(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")

    data_df = data_df[RELEVANT_COLUMNS]
    processor = DataProcessor(window_size=WINDOW_SIZE)
    X, y = processor.create_windows(data=data_df, coluna_alvo=TARGET_COLUMN, steps_ahead=STEPS_AHEAD)
    print(f"TREINO: Dimensões de X e y antes da normalização: X={X.shape}, y={y.shape}")
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)
    
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    print(f"TREINO: Dimensões de X_train_scaled: {X_train_scaled.shape}, y_train_scaled: {y_train_scaled.shape}")
    print(f"TREINO: Dimensões de X_validation_scaled: {X_validation_scaled.shape}, y_validation_scaled: {y_validation_scaled.shape}")
    print(f"TREINO: Dimensões de X_test_scaled: {X_test_scaled.shape}, y_test_scaled: {y_test_scaled.shape}")
    processor.save_scaler()

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainer(
        input_shape=input_shape,
        layers_config=[64, 32],
        dropout=0.2,
        batch_size=16,
        epochs=50,
        patience=5
    )

    lstm_model = trainer.train(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)
    trainer.save_model(lstm_model, model_path=MODEL_PATH)

    y_pred_scaled = lstm_model.predict(X_test_scaled)
    processor.load_scaler()
    print(f"TREINO: Dimensões de y_pred_scaled: {y_pred_scaled.shape}")
    y_pred = processor.inverse_transform(y_pred_scaled)
    y_test = processor.inverse_transform(y_test_scaled)
    print(f"TREINO: Dimensões de y_pred após inverse_transform: {y_pred.shape}")
    print(f"TREINO: Dimensões de y_test após inverse_transform: {y_test.shape}")
    timestamps = data_df.iloc[-len(y_test):]['timestamp'].values
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "real_value": y_test.flatten(),
        "predicted_value": y_pred.flatten()
    })

    results_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Resultados salvos em: {CSV_OUTPUT_PATH}")

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["timestamp"], results_df["real_value"], label="Real")
    plt.plot(results_df["timestamp"], results_df["predicted_value"], label="Previsto")
    plt.legend()
    plt.title("Previsão de Preço com LSTM")
    plt.xlabel("Timestamp")
    plt.ylabel("Preço")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()