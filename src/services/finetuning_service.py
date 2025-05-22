import os
import sys
import importlib
import pandas as pd
from datetime import datetime
import argparse
import json

src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None

from config.settings import Settings, set_seed
from database.model_binance import HourlyQuoteBitcoin
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter

set_seed(Settings.SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Fine-tuning (Keras, PyTorch, TensorFlow, LSTM, Transformer)")
    parser.add_argument("--framework", type=str, required=True, help="Framework: keras, pytorch ou tensorflow")
    parser.add_argument("--model_type", type=str, required=True, help="Tipo: lstm ou transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho do modelo salvo para fine-tuning.")
    parser.add_argument("--window_size", type=int, default=Settings.WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=Settings.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Settings.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=Settings.PATIENCE)
    parser.add_argument("--learning_rate", type=float, default=Settings.LEARNING_RATE)
    # Aqui, adicione outros argumentos universais se quiser
    return parser.parse_args()

def import_trainer_class(framework, model_type):
    module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"
    class_name = f"{framework.capitalize()}{model_type.capitalize()}Trainer"
    module = importlib.import_module(module_path)
    trainer_class = getattr(module, class_name)
    return trainer_class

def create_finetuning_folder(model_path):
    model_name = os.path.basename(model_path).replace(".h5", "").replace(".pt", "")
    base_path = os.path.join("result", "fine_tuning", model_name)
    timestamp_folder = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    full_path = os.path.join(base_path, timestamp_folder)
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(os.path.join(full_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "graficos"), exist_ok=True)
    return full_path

def save_finetuning_info(global_csv_path, base_path, **kwargs):
    file_exists = os.path.exists(global_csv_path)
    results = dict(kwargs)
    results["timestamp"] = kwargs.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["output_path"] = base_path
    results_df = pd.DataFrame([results])
    results_df.to_csv(global_csv_path, mode="a", header=not file_exists, index=False)

def universal_finetune(
    framework,
    model_type,
    model_path,
    window_size,
    epochs,
    batch_size,
    patience,
    learning_rate
    # Adicione outros hiperparâmetros aqui se quiser
):
    print(f"[Fine-tuning] Framework: {framework} | Model: {model_type} | Model Path: {model_path}")
    global_csv_path = os.path.join("result", "fine_tuning", "finetuning_results.csv")

    # Importa o trainer universal
    TrainerClass = import_trainer_class(framework, model_type)
    # O ideal é você ter um método `load_model` universal
    trainer = TrainerClass()
    model = trainer.load_model(model_path)

    # (Opcional) set optimizer/learning_rate de forma universal (pode colocar try/except para frameworks diferentes)
    if hasattr(trainer, "recompile"):
        trainer.recompile(model, learning_rate=learning_rate)
    elif hasattr(model, "compile"):
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=Settings.LOSS_FUNCTION,
            metrics=Settings.METRICS
        )

    # Carregar dados do banco
    data_df = HourlyQuoteBitcoin.get_between_dates(Settings.START_DATE, Settings.END_DATE)
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
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y, train_size=Settings.TRAIN_SIZE, validation_size=Settings.VALIDATION_SPLIT
    )
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    processor.save_scaler()

    # Treino universal
    if hasattr(trainer, "fine_tune"):
        history = trainer.fine_tune(
            model, X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=epochs, batch_size=batch_size, patience=patience
        )
        train_loss = history['loss'][-1]
        val_loss = history['val_loss'][-1]
    else:
        # Keras/TensorFlow padrão
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

    # Salva resultado e modelo
    base_path = create_finetuning_folder(model_path)
    # Use a extensão correta!
    ext = "h5" if framework.lower() == "keras" else "pt" if framework.lower() == "pytorch" else "model"
    model_save_path = os.path.join(base_path, "models", f"fine_tuned_model.{ext}")
    trainer.save_model(model, model_save_path)

    # Previsão e gráficos
    y_pred_scaled = trainer.predict(model, X_test_scaled)
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()

    save_finetuning_info(
        global_csv_path=global_csv_path,
        base_path=base_path,
        framework=framework,
        model_type=model_type,
        window_size=window_size,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        train_loss=train_loss,
        val_loss=val_loss,
        start_date=Settings.START_DATE,
        end_date=Settings.END_DATE,
        target_column=Settings.TARGET_COLUMN,
        relevant_columns=Settings.RELEVANT_COLUMNS,
        validation_split=Settings.VALIDATION_SPLIT,
        train_size=Settings.TRAIN_SIZE,
        steps_ahead=Settings.STEPS_AHEAD,
        gpu_used=Settings.USE_GPU,
        seed=Settings.SEED
    )
    # Salva previsões em CSV
    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=original_timestamps[-len(y_test):],
        y_test=y_test,
        y_pred=y_pred,
        output_path=os.path.join(base_path, "csv", "fine_tuning_results.csv")
    )

    # Gráficos
    plotter = Plotter()
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=os.path.join(base_path, "graficos", "price_predictions.png")
    )
    plotter.plot_errors_over_time(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
    )
    plotter.plot_histogram_of_errors(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "histogram_errors.png")
    )
    plotter.plot_scatter_real_vs_predicted(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "scatter_real_vs_predicted.png")
    )

    print(f"[Fine-tuning concluído] Modelo salvo em: {model_save_path}")

if __name__ == "__main__":
    args = parse_args()
    try:
        universal_finetune(
            framework=args.framework,
            model_type=args.model_type,
            model_path=args.model_path,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        print(f"[ERRO Fine-tuning]: {e}")
