import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from database.model_nahas import TrainingRun, FineTuningRun, db
from services.trainer_factory import TrainerFactory
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter
from config.settings import BASE_DIR, Settings, set_seed, set_cuda_tensorflow, set_cuda_pytorch
import random
from services.binance import BinanceData
from utils.db_utils import ensure_db_connection


def prepare_and_set_seed(config):
        # Usa a seed fornecida ou gera uma nova
        seed = config.get("seed", None)
        if seed is None:
            # Gera seed aleatória de 32 bits, já que TF, numpy, etc aceitam int32
            seed = random.SystemRandom().randint(0, 2**32 - 1)
            config["seed"] = seed  # Adiciona no config para ficar registrado!
            print(f"Seed não fornecida. Gerada: {config['seed']}")
        set_seed(seed)  # sua função já universaliza para numpy, tf, etc.
        return seed

def get_epochs_trained(model_or_history):
    # Caso seja Keras/TensorFlow com .history
    if hasattr(model_or_history, "history") and isinstance(model_or_history.history, dict):
        # keras==3 pode ser dict
        return len(model_or_history.history.get("loss", []))
    if hasattr(model_or_history, "history") and hasattr(model_or_history.history, "epoch"):
        return len(model_or_history.history.epoch)
    if hasattr(model_or_history, "epoch"):  # pytorch-lightning style
        return model_or_history.epoch if isinstance(model_or_history.epoch, int) else len(model_or_history.epoch)
    # Caso vc tenha customizado para salvar o número de epochs (PyTorch puro)
    if hasattr(model_or_history, "epochs_trained"):
        return model_or_history.epochs_trained
    # Se for o trainer PyTorch e você não salvou nada: retorna o número de epochs do config
    if hasattr(model_or_history, "epochs"):
        return model_or_history.epochs
    # Se for um dicionário de history
    if isinstance(model_or_history, dict):
        if "loss" in model_or_history:
            return len(model_or_history["loss"])
    return None


class ModelService:
    def __init__(self):
        pass

    def _get_save_dirs(self, base_folder, hash_id=None):
        """Cria estrutura de diretórios organizada na raiz do projeto (src/results/...)."""
        hash_id = hash_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Caminho base da pasta 'result' dentro da raiz do projeto
        base_path = os.path.join(BASE_DIR, "results", base_folder, hash_id)

        # Cria as subpastas necessárias
        os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scaler"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "hiperparams"), exist_ok=True)

        return base_path
    

    def universal_postprocess(self, y_pred, y_test, timestamps):
        # y_pred, y_test: np.array de shape [n, steps] ou [n,]
        # timestamps: array/lista de tamanho >= n (usar os últimos n)
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        timestamps = np.array(timestamps)

        # Garante sempre [n, steps]
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        n_samples = y_pred.shape[0]
        n_steps = y_pred.shape[1]
        y_test = y_test[:, :n_steps]  # Ajuste se vier steps extras

        # Checa shapes
        assert y_pred.shape == y_test.shape, f"Shape mismatch: y_test={y_test.shape}, y_pred={y_pred.shape}"
        assert len(timestamps) >= n_samples, "Timestamps length mismatch"

        return y_pred, y_test, timestamps[-n_samples:], n_steps

    def train(
        self, config: dict, framework: str, model_type: str, X=None, y=None, run_id=None
    ):
        """
        Treina qualquer modelo, qualquer framework, usando configuração.
        - config: dicionário de hiperparâmetros.
        - framework: 'keras', 'pytorch', 'tensorflow'
        - model_type: 'lstm', 'transformer', etc.
        - X, y: opcionais (se quiser passar dados prontos).
        """
        seed = prepare_and_set_seed(config)
        config["seed"] = seed
        if config.get("use_gpu", True):
            if framework.lower() in ("tensorflow", "keras"):
                set_cuda_tensorflow(config.get("gpu_index", 0))
            elif framework.lower() == "pytorch":
                set_cuda_pytorch(config.get("gpu_index", 0))
        hash_id = run_id or config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("train", hash_id)
        ext = {
            "keras": ".keras",
            "tensorflow": ".keras",
            "pytorch": ".pt"
        }.get(framework.lower(), ".model")

        model_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
        scaler_dir = os.path.join(base_path, "scaler")
        config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)  
        
        # 1. Dados
        if X is None or y is None:
            # Carregamento padrão (você pode parametrizar isso)
            from database.model_binance import HourlyQuoteBitcoin
            data_df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))
            if config.get("indicators_apply"):
                data_df = TechnicalIndicators.process_indicators(data_df, config.get("indicators_apply"))

            original_timestamps = data_df['timestamp'].values
            data_df = data_df[config.get("relevant_columns")]
            window_size = config.get("window_size")
            processor = DataProcessor(window_size=window_size)
            X, y = processor.create_windows(
                data=data_df, coluna_alvo=config.get("target_column"), steps_ahead=config.get("steps_ahead")
            )
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                X, y, train_size=config.get("train_size"), validation_size=config.get("validation_split")
            )
            X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
            processor.save_scaler(scaler_dir)
            X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
            X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
        else:
            # Treinamento com dados externos (para uso futuro)
            raise NotImplementedError("Suporte a dados externos ainda não implementado.")

        # 2. Cria trainer via Factory
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)

        # 3. Treina modelo
        model = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        trainer.save_model(model_path)
        # Métricas (Keras)
        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # 4. Predição/avaliação
        if hasattr(model, "predict"):
            y_pred_scaled = model.predict(X_test_scaled)
        elif hasattr(trainer, "predict"):
            y_pred_scaled = trainer.predict(model, X_test_scaled)
        else:
            raise RuntimeError("Seu trainer/modelo precisa de método predict")
        processor.load_scaler(scaler_dir)

        print("DEBUG y_pred_scaled shape:", y_pred_scaled.shape)
        print("DEBUG y_test_scaled shape:", y_test_scaled.shape)
        # Inversão e padronização
        y_pred = processor.inverse_transform(y_pred_scaled)
        y_test = processor.inverse_transform(y_test_scaled)
        y_pred, y_test, timestamps, n_steps = self.universal_postprocess(y_pred, y_test, original_timestamps)

        # Debug
        print("y_pred.shape:", y_pred.shape)
        print("y_test.shape:", y_test.shape)
        print("n_steps:", n_steps)
        print("Len timestamps:", len(timestamps))

        # 5. CSV + gráficos
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test, y_pred=y_pred, steps_ahead=n_steps, output_path=csv_path
        )
        plotter = Plotter()
        plotter.plot_price_predictions(
            results_df=results_df,
            timestamp_col="timestamp",
            real_col="real_value",
            pred_col="predicted_value",
            save_path=os.path.join(base_path, "graficos", "price_predictions.png")
        )
        plotter.plot_errors_over_time(
            y_test=y_test, y_pred=y_pred,
            save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
        )

        plotter.plot_correlation_matrix(
            df=data_df,
            columns=config.get("relevant_columns"),
            save_path=os.path.join(base_path, "graficos", "correlation_matrix.png")
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

         # 6. Salva no banco
        
            
        ensure_db_connection()
        TrainingRun.create(
            run_uuid=hash_id,
            start_time=config.get("start_date"),
            end_time=config.get("end_date"),
            status="finished",
            model_path=model_path,
            csv_metrics_path=csv_path,
            plot_dir=os.path.join(base_path, "graficos"),
            config_path=config_path,
            framework=framework,
            model_type=model_type,
            target_column=config.get("target_column"),
            seed=seed,         # ou Settings.SEED se preferir
            gpu_used=Settings.USE_GPU,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=get_epochs_trained(model if not hasattr(model, "history") else model.history),
            log=None,                           # log_msg pode ser None ou algum resumo do treino
        )
            
        # 6. Retorno
        return {
            "train_loss": float(train_loss) if train_loss else None,
            "val_loss": float(val_loss) if val_loss else None,
            "model_path": model_path,
            "csv_path": csv_path,
            "run_id": hash_id,
            "results_path": base_path,
        }

    def finetune(self, model_path, config, framework, model_type, original_run_id, run_id=None):
        """
        Fine-tuning universal + registro na tabela FineTuningRun.
        """

        hash_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("finetune", hash_id)
        if config.get("use_gpu", True):
            if framework.lower() in ("tensorflow", "keras"):
                set_cuda_tensorflow(config.get("gpu_index", 0))
            elif framework.lower() == "pytorch":
                set_cuda_pytorch(config.get("gpu_index", 0))

        ext = {
            "keras": ".keras",
            "tensorflow": ".keras",
            "pytorch": ".pt"
        }.get(framework.lower(), ".model")

        # Paths de salvamento
        model_ft_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
        scaler_dir = os.path.join(base_path, "scaler")
        config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
        plot_dir = os.path.join(base_path, "graficos")

        # Salva o config usado neste fine-tuning
        config["run_id"] = hash_id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)  

        # Carrega e processa dados (igual train)
        from database.model_binance import HourlyQuoteBitcoin
        data_df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))
        if config.get("indicators_apply"):
            data_df = TechnicalIndicators.process_indicators(data_df, config.get("indicators_apply"))
        print("DEBUG data_df shape:", data_df.head())
        original_timestamps = data_df['timestamp'].values
        data_df = data_df[config.get("relevant_columns")]
        window_size = config.get("window_size")
        processor = DataProcessor(window_size=window_size)
        X, y = processor.create_windows(
            data=data_df, coluna_alvo=config.get("target_column"), steps_ahead=config.get("steps_ahead")
        )
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X, y, train_size=config.get("train_size"), validation_size=config.get("validation_split")
        )
        X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
        processor.save_scaler(scaler_dir)
        X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
        X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

        # Instancia e carrega modelo antigo
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)
        trainer.load_model(model_path)

        # Fine-tune!
        model = trainer.finetune(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, config)
        trainer.save_model(model_ft_path)

        # Avaliação e métricas (igual ao train)
        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # Predição e inversão dos dados para gráficos/CSV
        if hasattr(model, "predict"):
            y_pred_scaled = model.predict(X_test_scaled)
        elif hasattr(trainer, "predict"):
            y_pred_scaled = trainer.predict(model, X_test_scaled)
        else:
            raise RuntimeError("Seu trainer/modelo precisa de método predict")
        processor.load_scaler(scaler_dir)
        y_pred = processor.inverse_transform(y_pred_scaled)
        y_test = processor.inverse_transform(y_test_scaled)
        y_pred, y_test, timestamps, n_steps = self.universal_postprocess(y_pred, y_test, original_timestamps)

        # Salva CSV + gráficos
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test, y_pred=y_pred, steps_ahead=n_steps, output_path=csv_path
        )
        plotter = Plotter()
        plotter.plot_price_predictions(
            results_df=results_df,
            timestamp_col="timestamp",
            real_col="real_value",
            pred_col="predicted_value",
            save_path=os.path.join(plot_dir, "price_predictions.png")
        )
        plotter.plot_errors_over_time(
            y_test=y_test, y_pred=y_pred,
            save_path=os.path.join(plot_dir, "errors_over_time.png")
        )
        plotter.plot_correlation_matrix(
            df=data_df,
            columns=config.get("relevant_columns"),
            save_path=os.path.join(plot_dir, "correlation_matrix.png")
        )
        plotter.plot_histogram_of_errors(
            y_test=y_test,
            y_pred=y_pred,
            save_path=os.path.join(plot_dir, "histogram_errors.png")
        )
        plotter.plot_scatter_real_vs_predicted(
            y_test=y_test,
            y_pred=y_pred,
            save_path=os.path.join(plot_dir, "scatter_real_vs_predicted.png")
        )

        # Salva métricas principais em JSON
        metrics_dict = {
            "train_loss": float(train_loss) if train_loss else None,
            "val_loss": float(val_loss) if val_loss else None,
            "n_steps": int(n_steps),
            "window_size": int(window_size),
            "run_id": hash_id,
            # ...adicione outras métricas relevantes!
        }
        metrics_json = json.dumps(metrics_dict, indent=4)

        # Salva registro do fine-tuning no banco
        ensure_db_connection()
        fine_tune_run = FineTuningRun.create(
            original_run=original_run_id,
            finetune_uuid=hash_id,
            status="finished",
            start_time=datetime.now(),  # ou pegue do início do processo
            end_time=datetime.now(),    # ou datetime ao terminar
            created_at=datetime.now(),
            updated_at=datetime.now(),
            finetuned_model_path=model_ft_path,
            finetune_config_path=config_path,
            finetune_csv_metrics_path=csv_path,
            finetune_plot_dir=plot_dir,
            metrics_json=metrics_json,
            framework=framework,
            model_type=model_type,
            seed=config.get("seed"),
            gpu_used=Settings.USE_GPU,
            train_loss=float(train_loss) if train_loss else None,
            val_loss=float(val_loss) if val_loss else None,
            best_epoch=int(model.history.epoch[-1]) if hasattr(model, "history") else None,
            log=None  # Pode preencher com algum log de fine-tune se quiser
        )

        return {
            "train_loss": float(train_loss) if train_loss else None,
            "val_loss": float(val_loss) if val_loss else None,
            "model_path": model_ft_path,
            "csv_path": csv_path,
            "run_id": hash_id,
            "results_path": base_path,
            "fine_tune_run_id": fine_tune_run.id,
            "metrics": metrics_dict,
        }

    
    def live_run_once(
        self,
        model_path: str,
        config_path: str,
        framework: str,
        model_type: str,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ):
        # 1. Carrega configs e define seed
        with open(config_path, "r") as f:
            config = json.load(f)
        seed = config.get("seed")
        if seed:
            set_seed(seed)
        window_size = config["window_size"]
        steps_ahead = config.get("steps_ahead", 1)
        relevant_columns = config["relevant_columns"]
        indicators_apply = config.get("indicators_apply", None)
        target_column = config["target_column"]

        # 2. Instancia trainer/model
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=(window_size, len(relevant_columns)), **config)
        model = trainer.load_model(model_path)
        processor = DataProcessor(window_size=window_size)
        scaler_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "scaler")
        processor.load_scaler(scaler_dir)

        # 3. Busca dados recentes da Binance
        

        binance = BinanceData()
        agora = datetime.utcnow()
        if agora.minute != 0 or agora.second != 0 or agora.microsecond != 0:
            agora -= timedelta(hours=1)
        delta = timedelta(minutes=agora.minute, seconds=agora.second, microseconds=agora.microsecond)
        end_time = agora - delta
        start_time = end_time - timedelta(hours=window_size)
        start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
        end_str = end_time.strftime("%d %b, %Y %H:%M:%S")
        df = binance.get_historical_data(symbol, start_str=start_str, interval=interval, end_str=end_str)

        # Indicadores técnicos (se houver)
        if indicators_apply:
            from utils.technical_indicators import TechnicalIndicators
            df = TechnicalIndicators.process_indicators(df, indicators_apply)

        for column in relevant_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

        filtered_columns = [col for col in relevant_columns if col != "close"]
        features = df[filtered_columns].values
        window_features = features[-window_size:]
        X_window = np.expand_dims(window_features, axis=0)
        X_normalized, _ = processor.apply_normalization(X_window, np.zeros((1,1)))
        X_input = X_normalized  # shape: (1, window_size, n_features)

        # Predição
        if hasattr(model, "predict"):
            y_pred_scaled = model.predict(X_input)
        elif hasattr(trainer, "predict"):
            y_pred_scaled = trainer.predict(model, X_input)
        else:
            raise RuntimeError("Trainer/modelo não possui método predict.")

        # Inverte normalização (usando processor universal)
        y_pred = processor.inverse_transform(y_pred_scaled)

        # Para multi-step, lista as previsões
        previsoes = [float(y_pred[0, i]) for i in range(steps_ahead)] if steps_ahead > 1 else [float(y_pred[0, 0])]

        return {
            "timestamp": str(df.iloc[-1]["timestamp"]) if "timestamp" in df.columns else str(datetime.utcnow()),
            "close_real": float(df.iloc[-1][target_column]),
            "previsoes": previsoes,
            "steps_ahead": steps_ahead,
            "target_column": target_column
        }