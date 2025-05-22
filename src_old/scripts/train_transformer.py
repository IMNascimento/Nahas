import os
import sys
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None

import argparse
import tensorflow as tf
from keras.callbacks import EarlyStopping
from config.settings import *
from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.transformer_model import CustomTransformerTrainer  # Importa seu transformer!
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter
from utils.technical_indicators import TechnicalIndicators
import hashlib
import time
import pandas as pd

set_seed(Settings.SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar modelo Transformer para predição financeira.")
    
    # Parâmetros gerais
    parser.add_argument("--window_size", type=int, default=Settings.WINDOW_SIZE, help="Tamanho da janela temporal.")

    # Hiperparâmetros do Transformer
    parser.add_argument("--num_layers", type=int, default=2, help="Nº de blocos Transformer (camadas encoder).")
    parser.add_argument("--d_model", type=int, default=64, help="Dimensão dos embeddings (d_model).")
    parser.add_argument("--num_heads", type=int, default=4, help="Nº de cabeças de atenção.")
    parser.add_argument("--dff", type=int, default=128, help="Dimensão feed-forward interno (dff).")
    parser.add_argument("--dropout_rate", type=float, default=Settings.DROPOUT, help="Dropout.")
    parser.add_argument("--batch_size", type=int, default=Settings.BATCH_SIZE, help="Tamanho do batch.")
    parser.add_argument("--epochs", type=int, default=Settings.EPOCHS, help="Épocas.")
    parser.add_argument("--patience", type=int, default=Settings.PATIENCE, help="Early stopping.")
    parser.add_argument("--loss", type=str, default=Settings.LOSS_FUNCTION, help="Função de perda.")
    parser.add_argument("--metrics", type=str, nargs="+", default=Settings.METRICS, help="Métricas.")
    parser.add_argument("--optimizer", type=str, default=Settings.OPTIMIZER, help="Otimizador ('adam').")
    parser.add_argument("--l1_reg", type=float, default=Settings.L1_REGULARIZATION, help="L1.")
    parser.add_argument("--l2_reg", type=float, default=Settings.L2_REGULARIZATION, help="L2.")
    parser.add_argument("--output_units", type=int, default=Settings.OUTPUT_UNITS, help="Saída.")
    parser.add_argument("--learning", type=float, default=Settings.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--activation", type=str, default=None, help="Ativação final ('linear', 'relu', etc).")

    return parser.parse_args()

def create_train_folder(config_hash):
    base_path = os.path.join("result/train", config_hash)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
    return base_path

def save_training_info(
    global_csv_path,
    window_size,
    num_layers,
    d_model,
    num_heads,
    dff,
    dropout_rate,
    batch_size,
    epochs,
    patience,
    loss_fn,
    metrics,
    optimizer,
    l1_reg,
    l2_reg,
    learning,
    output_units,
    train_loss,
    val_loss,
    model_path
):
    file_exists = os.path.exists(global_csv_path)
    results = {
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "dff": dff,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "window_size": window_size,
        "optimizer": optimizer,
        "loss_function": loss_fn,
        "metrics": metrics,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "learning_rate": learning,
        "output_units": output_units,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_path": model_path,
        "SEED": Settings.SEED,
        "COLUNAS": Settings.RELEVANT_COLUMNS,
        "ALVO": Settings.TARGET_COLUMN,
        "VALIDATION_SPLIT": Settings.VALIDATION_SPLIT,
        "TRAIN_SIZE": Settings.TRAIN_SIZE,
        "STEPS_AHEAD": Settings.STEPS_AHEAD,
        "GPU": Settings.USE_GPU,
        "END_DATE": Settings.END_DATE,
        "START_DATE": Settings.START_DATE
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(global_csv_path, mode="a", header=not file_exists, index=False)

def select_device():
    if Settings.USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("Treinando na GPU.")
            return tf.device('/GPU:0')
        else:
            print("GPU não disponível. Usando CPU.")
    else:
        print("Forçando uso da CPU.")
    return tf.device('/CPU:0')

def train_model(
    window_size,
    csv_output_path,
    model_path,
    plot_output_path,
    num_layers,
    d_model,
    num_heads,
    dff,
    dropout_rate,
    batch_size,
    epochs,
    patience,
    loss_fn,
    metrics,
    optimizer,
    l1_reg,
    l2_reg,
    learning,
    output_units,
    activation
):
    print(f"Treinando Transformer: layers={num_layers}, d_model={d_model}, heads={num_heads}, dff={dff}, dropout={dropout_rate}")

    global_csv_path = os.path.join("result/train", "training_results_transformer.csv")
    data_df = HourlyQuote.get_between_dates(Settings.START_DATE, Settings.END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados.")

    data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[Settings.RELEVANT_COLUMNS]
    processor = DataProcessor(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=Settings.TARGET_COLUMN,
        steps_ahead=Settings.STEPS_AHEAD
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(
        X, y, train_size=Settings.TRAIN_SIZE, validation_size=Settings.VALIDATION_SPLIT
    )

    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    processor.save_scaler()
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    trainer = CustomTransformerTrainer(
        input_shape=input_shape,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        output_units=output_units,
        learning_rate=learning,
        activation=activation
    )

    X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    y_train_scaled = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
    X_validation_scaled = tf.convert_to_tensor(X_validation_scaled, dtype=tf.float32)
    y_validation_scaled = tf.convert_to_tensor(y_validation_scaled, dtype=tf.float32)

    with select_device():
        model = trainer.train(
            X_train_scaled, y_train_scaled,
            X_validation_scaled, y_validation_scaled
        )
    trainer.save_model(model, model_path=model_path)

    # history está dentro do model.history no Keras Functional API
    history = model.history
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    save_training_info(
        global_csv_path,
        window_size,
        num_layers,
        d_model,
        num_heads,
        dff,
        dropout_rate,
        batch_size,
        epochs,
        patience,
        loss_fn,
        metrics,
        optimizer,
        l1_reg,
        l2_reg,
        learning,
        output_units,
        train_loss,
        val_loss,
        model_path
    )

    # Previsão
    X_test_scaled = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
    y_pred_scaled = model.predict(X_test_scaled)
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]
    assert len(timestamps) == len(y_test) == len(y_pred)

    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=timestamps,
        y_test=y_test,
        y_pred=y_pred,
        output_path=csv_output_path
    )

    plotter = Plotter()
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=f"{plot_output_path}price_predictions.png"
    )

    plotter.plot_correlation_matrix(
        df=data_df,
        columns=Settings.RELEVANT_COLUMNS,
        save_path=f"{plot_output_path}correlation_matrix.png"
    )

    plotter.plot_errors_over_time(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}errors_over_time.png"
    )

    plotter.plot_histogram_of_errors(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}histogram_errors.png"
    )

    plotter.plot_scatter_real_vs_predicted(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}scatter_real_vs_predicted.png"
    )

    print(f"Modelo Transformer salvo em: {model_path}")
    print(f"Resultados salvos em: {csv_output_path}")

if __name__ == "__main__":
    args = parse_args()
    unique_input = str(args) + str(time.time())
    config_hash = hashlib.md5(unique_input.encode()).hexdigest()[:8]
    train_folder = create_train_folder(config_hash)

    model_path = os.path.join(train_folder, "models", f"transformer_{config_hash}.h5")
    csv_output_path = os.path.join(train_folder, "csv", f"results_{config_hash}.csv")
    plot_output_path = os.path.join(train_folder, "graficos/")

    try:
        train_model(
            window_size=args.window_size,
            csv_output_path=csv_output_path,
            model_path=model_path,
            plot_output_path=plot_output_path,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            loss_fn=args.loss,
            metrics=args.metrics,
            optimizer=args.optimizer,
            l1_reg=args.l1_reg,
            l2_reg=args.l2_reg,
            learning=args.learning,
            output_units=args.output_units,
            activation=args.activation
        )
    except Exception as e:
        print(f"Erro ao treinar o modelo Transformer: {e}")
