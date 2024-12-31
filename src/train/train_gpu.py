import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from keras.callbacks import EarlyStopping
from models.db.model_binance import HourlyQuote
from data.data_gpu_processing import DataProcessorGPU
from models.lstm_gpu import CustomLSTMTrainerGPU
from config.settings import Settings
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter

# Configurações
END_DATE = "2024-09-14 23:59:59"



def train_model(
    dropout, batch_size, epochs, patience, layers_config,
    window_size, csv_output_path, model_path, plot_output_path=None
):
    print(f"Treinando modelo com DROPOUT={dropout}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, LAYERS={layers_config}")
    data_df = HourlyQuote.get_to_date(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[Settings.RELEVANT_COLUMNS]
    processor = DataProcessorGPU(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=Settings.TARGET_COLUMN,
        steps_ahead=Settings.STEPS_AHEAD
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)
    # Normalização
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    # Configuração do modelo
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainerGPU(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )
    # Converte tudo para float32 antes de treinar (caso seja NumPy)
    X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    y_train_scaled = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
    X_validation_scaled = tf.convert_to_tensor(X_validation_scaled, dtype=tf.float32)
    y_validation_scaled = tf.convert_to_tensor(y_validation_scaled, dtype=tf.float32)
    
    # Força o uso de apenas 1 GPU, se disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        device_to_use = "/GPU:0"
        print("Treinando no dispositivo:", device_to_use)
    else:
        device_to_use = "/CPU:0"
        print("Nenhuma GPU encontrada. Treinando em CPU.")
    with tf.device(device_to_use):
        # Treina o modelo
        lstm_model = trainer.train(
            X_train_scaled, y_train_scaled,
            X_validation_scaled, y_validation_scaled
        )
    
    # Salva o modelo
    trainer.save_model(lstm_model, model_path=model_path)
    # Previsão
    X_test_scaled = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
    y_pred_scaled = lstm_model.predict(X_test_scaled)
    # Carrega scalers e inverte normalização
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled.numpy()).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]
    assert len(timestamps) == len(y_test) == len(y_pred), (
        f"Dimensões incompatíveis: timestamps({len(timestamps)}), "
        f"y_test({len(y_test)}), y_pred({len(y_pred)})"
    )
    
    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=timestamps,
        y_test=y_test,
        y_pred=y_pred,
        output_path=csv_output_path
    )

    plotter = Plotter()
    # Plot de previsões
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=plot_output_path
    )

    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {csv_output_path}")


if __name__ == "__main__":

    csv_output_path = f"result/csv/treinos/treino_dropout_{Settings.DROPOUT}_batch_{Settings.BATCH_SIZE}_window_{Settings.WINDOW_SIZE}_layers_{Settings.LAYERS_CONFIG}.csv"
    model_path = f"result/models/model_dropout_{Settings.DROPOUT}_batch_{Settings.BATCH_SIZE}_window_{Settings.WINDOW_SIZE}_layers_{Settings.LAYERS_CONFIG}.h5"
    plot_output_path = f"result/graficos/treinos/model_dropout_{Settings.DROPOUT}_batch_{Settings.BATCH_SIZE}_window_{Settings.WINDOW_SIZE}_layers_{Settings.LAYERS_CONFIG}.png"
    
    try:
        train_model(
            dropout=Settings.DROPOUT,
            batch_size=Settings.BATCH_SIZE,
            epochs=Settings.EPOCHS,
            patience=Settings.PATIENCE,
            layers_config=Settings.LAYERS_CONFIG,
            window_size=Settings.WINDOW_SIZE,
            csv_output_path=csv_output_path,
            model_path=model_path,
            plot_output_path=plot_output_path
        )
    except Exception as e:
        print(f"Erro ao treinar o modelo: {e}")