import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import random

from database.model_nahas import TrainingRun, FineTuningRun, db
from services.trainer_factory import TrainerFactory
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter
from config.settings import BASE_DIR, Settings, set_seed, set_cuda_tensorflow, set_cuda_pytorch
from services.binance import BinanceData
from utils.db_utils import ensure_db_connection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.evomsn_normalizer import EvoMSNNormalizer, EvoMSNLikeNormalizer


# -------------------------
# helpers (seed/epochs)
# -------------------------
def prepare_and_set_seed(config: dict) -> int:
    seed = config.get("seed")
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        config["seed"] = seed
        print(f"Seed não fornecida. Gerada: {seed}")
    set_seed(seed)
    return seed

def get_epochs_trained(model_or_history):
    if hasattr(model_or_history, "history") and isinstance(model_or_history.history, dict):
        return len(model_or_history.history.get("loss", []))
    if hasattr(model_or_history, "history") and hasattr(model_or_history.history, "epoch"):
        return len(model_or_history.history.epoch)
    if hasattr(model_or_history, "epoch"):
        return model_or_history.epoch if isinstance(model_or_history.epoch, int) else len(model_or_history.epoch)
    if hasattr(model_or_history, "epochs_trained"):
        return model_or_history.epochs_trained
    if hasattr(model_or_history, "epochs"):
        return model_or_history.epochs
    if isinstance(model_or_history, dict) and "loss" in model_or_history:
        return len(model_or_history["loss"])
    return None

def _parse_dt(x):
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        # mesmo formato que você grava na config
        return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    return datetime.now()


class ModelService:
    def __init__(self):
        pass

    # -------------------------
    # métricas básicas
    # -------------------------
    @staticmethod
    def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        r2  = float(r2_score(yt, yp))
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    # -------------------------
    # infra de pastas
    # -------------------------
    def _get_save_dirs(self, base_folder, hash_id=None):
        hash_id = hash_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(BASE_DIR, "results", base_folder, hash_id)
        os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scaler"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "hiperparams"), exist_ok=True)
        return base_path

    # -------------------------
    # pós-processamento universal
    # -------------------------
    def universal_postprocess(self, y_pred, y_test, timestamps):
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        timestamps = np.array(timestamps)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        n_samples = y_pred.shape[0]
        n_steps = y_pred.shape[1]
        y_test = y_test[:, :n_steps]

        assert y_pred.shape == y_test.shape, f"Shape mismatch: y_test={y_test.shape}, y_pred={y_pred.shape}"
        assert len(timestamps) >= n_samples, "Timestamps length mismatch"

        return y_pred, y_test, timestamps[-n_samples:], n_steps

    # -------------------------
    # util: extrair config de normalização
    # -------------------------
    def _norm_cfg(self, config: dict):
        norm = config.get("normalization", {}) or {}
        strategy = norm.get("strategy", "global").lower()  # "global" | "local" | "evomsn" | "evomsn_like"
        if strategy not in ("global", "local", "evomsn", "evomsn_like"):
            raise ValueError("normalization.strategy deve ser 'global', 'local', 'evomsn' ou 'evomsn_like'.")

        scaler_type = norm.get("scaler_type", "robust")   # global
        x_mode = norm.get("x_mode", "zscore")             # local
        y_mode = norm.get("y_mode", "none")               # local

        # EvoMSN params
        ev_k   = int(norm.get("evomsn_k_scales", config.get("evomsn_k_scales", 4)))
        ev_agg = norm.get("evomsn_agg", "fft")            # "fft" | "uniform"
        ev_pred = norm.get("evomsn_predictor", config.get("evomsn_predictor", "linear"))  # "linear" | "mlp"

        return strategy, scaler_type, x_mode, y_mode, {"k_scales": ev_k, "agg": ev_agg, "predictor": ev_pred}

    # -------------------------
    # util: y_mode local exige o alvo dentro do X
    # -------------------------
    @staticmethod
    def _y_mode_needs_target_in_X(y_mode: str) -> bool:
        return y_mode in ("relative_last", "zscore_target", "minmax_target", "robust_target")

    # constrói janelas do alvo para concatenar como último canal do X
    @staticmethod
    def _build_target_windows(target_series: np.ndarray, window_size: int, steps_ahead: int) -> np.ndarray:
        n = len(target_series) - window_size - steps_ahead + 1
        if n <= 0:
            raise ValueError("Série muito curta para window_size e steps_ahead.")
        tw = np.empty((n, window_size, 1), dtype=float)
        for i in range(n):
            tw[i, :, 0] = target_series[i:i + window_size]
        return tw

    # split timestamps na mesma lógica do split_data
    @staticmethod
    def _split_timestamps(ts_all: np.ndarray, train_size: float, val_size: float):
        n_total = len(ts_all)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)
        ts_train = ts_all[:n_train]
        ts_val = ts_all[n_train:n_train + n_val]
        ts_test = ts_all[n_train + n_val:]
        return ts_train, ts_val, ts_test

    # -------------------------
    # TREINO
    # -------------------------
    def train(self, config: dict, framework: str, model_type: str, X=None, y=None, run_id=None):
        seed = prepare_and_set_seed(config)
        config["seed"] = seed

        if config.get("use_gpu", True):
            if framework.lower() in ("tensorflow", "keras"):
                set_cuda_tensorflow(config.get("gpu_index", 0))
            elif framework.lower() == "pytorch":
                set_cuda_pytorch(config.get("gpu_index", 0))

        norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)
        hash_id = run_id or config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("train", hash_id)
        ext = {"keras": ".keras", "tensorflow": ".keras", "pytorch": ".pt"}.get(framework.lower(), ".model")

        model_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
        scaler_dir = os.path.join(base_path, "scaler")
        config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
        metrics_path = os.path.join(base_path, "metrics.json")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # -------------------------
        # 1) Dados
        # -------------------------
        if X is None or y is None:
            from database.model_binance import HourlyQuoteBitcoin
            df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))

            if config.get("indicators_apply"):
                df = TechnicalIndicators.process_indicators(df, config.get("indicators_apply"))

            for c in config.get("relevant_columns", []):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            window_size = int(config.get("window_size"))
            steps_ahead = int(config.get("steps_ahead", 1))
            target_col = config.get("target_column")
            assert target_col in df.columns, f"Coluna alvo '{target_col}' não encontrada em df."

            feat_cols = [c for c in (config.get("relevant_columns") or []) if c != "timestamp"]
            cols_for_window = list(dict.fromkeys(feat_cols + [target_col, "timestamp"]))
            data_df = df[cols_for_window].dropna().reset_index(drop=True)

            processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)

            # janelas + timestamps alinhados ao horizonte
            X_all, y_all, ts_all = processor.create_windows_and_timestamps(
                data_df, coluna_alvo=target_col, steps_ahead=steps_ahead,
                timestamp_col="timestamp", ts_mode="horizon"
            )

            # Para estratégias que precisam do alvo no X (local com certos y_mode)
            if norm_strategy == "local" and self._y_mode_needs_target_in_X(y_mode):
                target_series = data_df[target_col].values
                tw_all = self._build_target_windows(target_series, window_size, steps_ahead)  # (n, L, 1)
                X_all = np.concatenate([X_all, tw_all], axis=2)
            else:
                # usado pelo EvoMSN (pesos do ensemble via janela do alvo)
                tw_all = self._build_target_windows(data_df[target_col].values, window_size, steps_ahead)

            # split
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                X_all, y_all,
                train_size=float(config.get("train_size", 0.7)),
                validation_size=float(config.get("validation_split", 0.15))
            )
            ts_train, ts_val, ts_test = self._split_timestamps(
                ts_all, float(config.get("train_size", 0.7)), float(config.get("validation_split", 0.15))
            )
            # target windows alinhadas (para pesos do ensemble)
            n_tr, n_v = X_train.shape[0], X_val.shape[0]
            tw_train, tw_val, tw_test = tw_all[:n_tr], tw_all[n_tr:n_tr + n_v], tw_all[n_tr + n_v:]

            # -------------------------
            # normalização por estratégia
            # -------------------------
            ctx_train_local = None   # usado só se local
            ctx_train_msn   = None   # usado só se evomsn/like

            if norm_strategy == "global":
                X_train_scaled, y_train_scaled = processor.normalize_global(X_train, y_train)
                processor.save_scaler(scaler_dir)
                X_val_scaled,  y_val_scaled  = processor.apply_normalization_global(X_val,  y_val)
                X_test_scaled, y_test_scaled = processor.apply_normalization_global(X_test, y_test)
                inverse_kind = "global"
                inverse_ctx_test = None

            elif norm_strategy == "local":
                target_idx = X_train.shape[2] - 1 if self._y_mode_needs_target_in_X(y_mode) else 0
                # guarde ctx do TREINO para inversão de métricas do treino
                X_train_scaled, y_train_scaled, ctx_train_local = processor.normalize_local(
                    X_train, y_train, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_val_scaled, y_val_scaled, _ = processor.apply_normalization_local(
                    X_val, y_val, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_test_scaled, y_test_scaled, inverse_ctx_test = processor.apply_normalization_local(
                    X_test, y_test, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                inverse_kind = "local"

            else:  # "evomsn" ou "evomsn_like"
                if norm_strategy == "evomsn":
                    msn = EvoMSNNormalizer(
                        window_size=window_size,
                        horizon=steps_ahead,
                        n_features=X_train.shape[2],
                        target_idx=0,  # pesos vêm de target_windows
                        k_scales=int(evcfg["k_scales"]),
                        agg=str(evcfg.get("agg", "fft")),
                        random_state=seed,
                        predictor_type=str(evcfg.get("predictor", "linear")),
                    )
                else:
                    msn = EvoMSNLikeNormalizer(
                        window_size=window_size,
                        horizon=steps_ahead,
                        n_features=X_train.shape[2],
                        target_idx=0,
                        k_scales=int(evcfg["k_scales"]),
                        agg=str(evcfg.get("agg", "fft")),
                        random_state=seed,
                    )

                # treino MSN (períodos globais a partir de X_train)
                X_train_scaled, y_train_scaled = msn.fit(X_train, y_train, save_path=scaler_dir)
                # obtenha ctx também para o CONJUNTO DE TREINO (para inversão das métricas)
                _Xtr_tmp, _ytr_tmp, ctx_train_msn = msn.transform(X_train, None, target_windows=tw_train)
                # validação e teste com pesos via janela do alvo
                X_val_scaled,  y_val_scaled,  _ctx_val  = msn.transform(X_val,  y_val,  target_windows=tw_val)
                X_test_scaled, _y_dummy,     inverse_ctx_test = msn.transform(X_test, None,   target_windows=tw_test)
                inverse_kind = "evomsn"

                # empacote objeto e ctx de teste para a etapa de inversão
                inverse_ctx_test = (msn, inverse_ctx_test)

        else:
            raise NotImplementedError("Treino com X,y externos ainda não implementado.")

        # -------------------------
        # 2) Trainer/Modelo
        # -------------------------
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)

        # -------------------------
        # 3) Treinar
        # -------------------------
        model = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        trainer.save_model(model_path)

        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # -------------------------
        # 4A) Predição + inversão (TESTE)
        # -------------------------
        if hasattr(model, "predict"):
            y_pred_scaled_test = model.predict(X_test_scaled)
        elif hasattr(trainer, "predict"):
            y_pred_scaled_test = trainer.predict(model, X_test_scaled)
        else:
            raise RuntimeError("Seu trainer/modelo precisa de método predict.")

        if inverse_kind == "global":
            y_pred_test = processor.inverse_transform_global(y_pred_scaled_test)
            y_test_orig = processor.inverse_transform_global(y_test_scaled)

        elif inverse_kind == "local":
            y_pred_test = processor.inverse_transform_local(y_pred_scaled_test, inverse_ctx_test)
            y_test_orig = processor.inverse_transform_local(y_test_scaled, inverse_ctx_test)

        else:  # "evomsn" e "evomsn_like"
            msn, ctx_test = inverse_ctx_test
            _per_scale_test, y_pred_test = msn.denorm_and_ensemble(y_pred_scaled_test, ctx_test)  # (N,H)
            y_test_orig = y_test  # já está no domínio real nesse caminho

        # alinhar shapes/timestamps (TESTE)
        y_pred_test, y_test_orig, timestamps, n_steps = self.universal_postprocess(y_pred_test, y_test_orig, ts_test)

        # -------------------------
        # 4B) Predição + inversão (TREINO) para métricas
        # -------------------------
        if hasattr(model, "predict"):
            y_pred_scaled_train = model.predict(X_train_scaled)
        else:
            y_pred_scaled_train = trainer.predict(model, X_train_scaled)

        if inverse_kind == "global":
            y_pred_train = processor.inverse_transform_global(y_pred_scaled_train)
            y_train_orig = processor.inverse_transform_global(y_train_scaled)

        elif inverse_kind == "local":
            # usa o ctx do TREINO que guardamos na normalização local
            y_pred_train = processor.inverse_transform_local(y_pred_scaled_train, ctx_train_local)
            y_train_orig = processor.inverse_transform_local(y_train_scaled, ctx_train_local)

        else:  # "evomsn"/"evomsn_like"
            # ctx de treino calculado acima (ctx_train_msn)
            _per_scale_tr, y_pred_train = msn.denorm_and_ensemble(y_pred_scaled_train, ctx_train_msn)
            y_train_orig = y_train  # já no domínio real

        # alinhar shapes para métricas de treino (timestamps não usados nas métricas):
        y_pred_train = np.array(y_pred_train)
        y_train_orig = np.array(y_train_orig)
        if y_pred_train.ndim == 1:
            y_pred_train = y_pred_train.reshape(-1, 1)
        if y_train_orig.ndim == 1:
            y_train_orig = y_train_orig.reshape(-1, 1)
        # garantir colunas compatíveis
        n_steps_train = y_pred_train.shape[1]
        y_train_orig = y_train_orig[:, :n_steps_train]

        # -------------------------
        # 4C) Métricas (Train/Test)
        # -------------------------
        metrics_train = self._compute_basic_metrics(y_train_orig, y_pred_train)
        metrics_test  = self._compute_basic_metrics(y_test_orig, y_pred_test)

        metrics_readable = {
            "MAE - Train data": metrics_train["mae"],
            "MAE - Test data": metrics_test["mae"],
            "RMSE - Train data": metrics_train["rmse"],
            "RMSE - Test data": metrics_test["rmse"],
            "MSE - Train data": metrics_train["mse"],
            "MSE - Test data": metrics_test["mse"],
            "R2 score - Train data": metrics_train["r2"],
            "R2 score - Test data": metrics_test["r2"],
        }
        metrics_bundle = {
            "train": metrics_train,
            "test": metrics_test,
            "readable": metrics_readable,
            "n_steps": int(n_steps),
            "window_size": int(window_size),
            "run_id": hash_id,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_bundle, f, indent=4)

        # -------------------------
        # 5) CSV + Gráficos (TESTE)
        # -------------------------
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test_orig, y_pred=y_pred_test,
            steps_ahead=n_steps, output_path=csv_path
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
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
        )
        try:
            plotter.plot_correlation_matrix(
                df=df,
                columns=config.get("relevant_columns"),
                save_path=os.path.join(base_path, "graficos", "correlation_matrix.png")
            )
        except Exception:
            pass
        plotter.plot_histogram_of_errors(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "histogram_errors.png")
        )
        plotter.plot_scatter_real_vs_predicted(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "scatter_real_vs_predicted.png")
        )

        # -------------------------
        # 6) Banco
        # -------------------------
        metrics_json_db = json.dumps(metrics_bundle, indent=4)
        ensure_db_connection()
        TrainingRun.create(
            run_uuid=hash_id,
            start_time=_parse_dt(config.get("start_date")),
            end_time=_parse_dt(config.get("end_date")),
            status="finished",
            model_path=model_path,
            csv_metrics_path=csv_path,
            plot_dir=os.path.join(base_path, "graficos"),
            config_path=config_path,
            framework=framework,
            model_type=model_type,
            target_column=config.get("target_column"),
            seed=seed,
            gpu_used=Settings.USE_GPU,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=get_epochs_trained(model if not hasattr(model, "history") else model.history),
            log=None,
            metrics_json=metrics_json_db, 
        )

        return {
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "model_path": model_path,
            "csv_path": csv_path,
            "metrics": metrics_bundle,
            "metrics_path": metrics_path,
            "run_id": hash_id,
            "results_path": base_path,
        }

    # -------------------------
    # FINE-TUNE
    # -------------------------
    def finetune(self, model_path, config, framework, model_type, original_run_id, run_id=None):
        norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)

        hash_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("finetune", hash_id)

        if config.get("use_gpu", True):
            if framework.lower() in ("tensorflow", "keras"):
                set_cuda_tensorflow(config.get("gpu_index", 0))
            elif framework.lower() == "pytorch":
                set_cuda_pytorch(config.get("gpu_index", 0))

        ext = {"keras": ".keras", "tensorflow": ".keras", "pytorch": ".pt"}.get(framework.lower(), ".model")
        model_ft_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
        scaler_dir = os.path.join(base_path, "scaler")
        config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
        plot_dir = os.path.join(base_path, "graficos")
        metrics_path = os.path.join(base_path, "metrics.json")

        config["run_id"] = hash_id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # dados
        from database.model_binance import HourlyQuoteBitcoin
        df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))
        if config.get("indicators_apply"):
            df = TechnicalIndicators.process_indicators(df, config.get("indicators_apply"))

        for c in config.get("relevant_columns", []):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        window_size = int(config.get("window_size"))
        steps_ahead = int(config.get("steps_ahead", 1))
        target_col = config.get("target_column")

        feat_cols = [c for c in (config.get("relevant_columns") or []) if c != "timestamp"]
        cols_for_window = list(dict.fromkeys(feat_cols + [target_col, "timestamp"]))
        data_df = df[cols_for_window].dropna().reset_index(drop=True)

        processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)
        X_all, y_all, ts_all = processor.create_windows_and_timestamps(
            data_df, coluna_alvo=target_col, steps_ahead=steps_ahead,
            timestamp_col="timestamp", ts_mode="horizon"
        )
        tw_all = self._build_target_windows(data_df[target_col].values, window_size, steps_ahead)

        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X_all, y_all,
            train_size=float(config.get("train_size", 0.7)),
            validation_size=float(config.get("validation_split", 0.15))
        )
        ts_train, ts_val, ts_test = self._split_timestamps(
            ts_all, float(config.get("train_size", 0.7)), float(config.get("validation_split", 0.15))
        )
        n_tr, n_v = X_train.shape[0], X_val.shape[0]
        tw_train, tw_val, tw_test = tw_all[:n_tr], tw_all[n_tr:n_tr + n_v], tw_all[n_tr + n_v:]

        # normalização
        ctx_train_local = None
        ctx_train_msn = None

        if norm_strategy == "global":
            X_train_scaled, y_train_scaled = processor.normalize_global(X_train, y_train)
            processor.save_scaler(scaler_dir)
            X_val_scaled,  y_val_scaled  = processor.apply_normalization_global(X_val,  y_val)
            X_test_scaled, y_test_scaled = processor.apply_normalization_global(X_test, y_test)
            inverse_kind = "global"
            inverse_ctx_test = None

        elif norm_strategy == "local":
            target_idx = 0  # aqui não usamos alvo dentro do X no fine-tune por padrão
            X_train_scaled, y_train_scaled, ctx_train_local = processor.normalize_local(
                X_train, y_train, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
            )
            X_val_scaled, y_val_scaled, _ = processor.apply_normalization_local(
                X_val, y_val, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
            )
            X_test_scaled, y_test_scaled, inverse_ctx_test = processor.apply_normalization_local(
                X_test, y_test, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
            )
            inverse_kind = "local"

        else:
            if norm_strategy == "evomsn":
                msn = EvoMSNNormalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=X_train.shape[2],
                    target_idx=0,
                    k_scales=int(evcfg["k_scales"]),
                    agg=str(evcfg.get("agg", "fft")),
                    random_state=config.get("seed", 42),
                    predictor_type=str(evcfg.get("predictor", "linear")),
                )
            else:
                msn = EvoMSNLikeNormalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=X_train.shape[2],
                    target_idx=0,
                    k_scales=int(evcfg["k_scales"]),
                    agg=str(evcfg.get("agg", "fft")),
                    random_state=config.get("seed", 42),
                )
            X_train_scaled, y_train_scaled = msn.fit(X_train, y_train, save_path=scaler_dir)
            _Xtr_tmp, _ytr_tmp, ctx_train_msn = msn.transform(X_train, None, target_windows=tw_train)
            X_val_scaled,  y_val_scaled,  _ctx_val  = msn.transform(X_val,  y_val,  target_windows=tw_val)
            X_test_scaled, _y_dummy,     ctx_test = msn.transform(X_test, None,   target_windows=tw_test)
            inverse_kind = "evomsn"
            inverse_ctx_test = (msn, ctx_test)

        # trainer/modelo
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)
        trainer.load_model(model_path)

        # fine-tune
        model = trainer.finetune(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, config)
        trainer.save_model(model_ft_path)

        # métricas (loss)
        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # predição TESTE + inversão
        if hasattr(model, "predict"):
            y_pred_scaled_test = model.predict(X_test_scaled)
        else:
            y_pred_scaled_test = trainer.predict(model, X_test_scaled)

        if inverse_kind == "global":
            y_pred_test = processor.inverse_transform_global(y_pred_scaled_test)
            y_test_orig = processor.inverse_transform_global(y_test_scaled)
        elif inverse_kind == "local":
            y_pred_test = processor.inverse_transform_local(y_pred_scaled_test, inverse_ctx_test)
            y_test_orig = processor.inverse_transform_local(y_test_scaled, inverse_ctx_test)
        else:
            msn, ctx_test = inverse_ctx_test
            _per_scale_te, y_pred_test = msn.denorm_and_ensemble(y_pred_scaled_test, ctx_test)
            y_test_orig = y_test

        y_pred_test, y_test_orig, timestamps, n_steps = self.universal_postprocess(y_pred_test, y_test_orig, ts_test)

        # predição TREINO + inversão (para métricas)
        if hasattr(model, "predict"):
            y_pred_scaled_train = model.predict(X_train_scaled)
        else:
            y_pred_scaled_train = trainer.predict(model, X_train_scaled)

        if inverse_kind == "global":
            y_pred_train = processor.inverse_transform_global(y_pred_scaled_train)
            y_train_orig = processor.inverse_transform_global(y_train_scaled)
        elif inverse_kind == "local":
            y_pred_train = processor.inverse_transform_local(y_pred_scaled_train, ctx_train_local)
            y_train_orig = processor.inverse_transform_local(y_train_scaled, ctx_train_local)
        else:
            _per_scale_tr, y_pred_train = msn.denorm_and_ensemble(y_pred_scaled_train, ctx_train_msn)
            y_train_orig = y_train

        # alinhar shapes para métricas de treino
        y_pred_train = np.array(y_pred_train)
        y_train_orig = np.array(y_train_orig)
        if y_pred_train.ndim == 1:
            y_pred_train = y_pred_train.reshape(-1, 1)
        if y_train_orig.ndim == 1:
            y_train_orig = y_train_orig.reshape(-1, 1)
        y_train_orig = y_train_orig[:, :y_pred_train.shape[1]]

        # métricas (Train/Test)
        metrics_train = self._compute_basic_metrics(y_train_orig, y_pred_train)
        metrics_test  = self._compute_basic_metrics(y_test_orig, y_pred_test)
        metrics_readable = {
            "MAE - Train data": metrics_train["mae"],
            "MAE - Test data": metrics_test["mae"],
            "RMSE - Train data": metrics_train["rmse"],
            "RMSE - Test data": metrics_test["rmse"],
            "MSE - Train data": metrics_train["mse"],
            "MSE - Test data": metrics_test["mse"],
            "R2 score - Train data": metrics_train["r2"],
            "R2 score - Test data": metrics_test["r2"],
        }
        metrics_bundle = {
            "train": metrics_train,
            "test": metrics_test,
            "readable": metrics_readable,
            "n_steps": int(n_steps),
            "window_size": int(window_size),
            "run_id": hash_id,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_bundle, f, indent=4)

        # artefatos (TESTE)
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test_orig, y_pred=y_pred_test,
            steps_ahead=n_steps, output_path=csv_path
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
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(plot_dir, "errors_over_time.png")
        )
        try:
            plotter.plot_correlation_matrix(
                df=df, columns=config.get("relevant_columns"),
                save_path=os.path.join(plot_dir, "correlation_matrix.png")
            )
        except Exception:
            pass
        plotter.plot_histogram_of_errors(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(plot_dir, "histogram_errors.png")
        )
        plotter.plot_scatter_real_vs_predicted(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(plot_dir, "scatter_real_vs_predicted.png")
        )

        # banco
        metrics_dict = {
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "n_steps": int(n_steps),
            "window_size": int(window_size),
            "run_id": hash_id,
        }
        metrics_json = json.dumps(metrics_dict, indent=4)

        ensure_db_connection()
        fine_tune_run = FineTuningRun.create(
            original_run=original_run_id,
            finetune_uuid=hash_id,
            status="finished",
            start_time=datetime.now(),
            end_time=datetime.now(),
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
            train_loss=float(train_loss) if train_loss is not None else None,
            val_loss=float(val_loss) if val_loss is not None else None,
            best_epoch=int(model.history.epoch[-1]) if hasattr(model, "history") else None,
            log=None
        )

        return {
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "model_path": model_ft_path,
            "csv_path": csv_path,
            "metrics": metrics_bundle,
            "metrics_path": metrics_path,
            "run_id": hash_id,
            "results_path": base_path,
            "fine_tune_run_id": fine_tune_run.id,
        }

    # -------------------------
    # LIVE RUN (único ciclo)
    # -------------------------
    def live_run_once(
        self,
        model_path: str,
        config_path: str,
        framework: str,
        model_type: str,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        seed = config.get("seed")
        if seed:
            set_seed(seed)

        norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)
        window_size = int(config["window_size"])
        steps_ahead = int(config.get("steps_ahead", 1))
        relevant_columns = config["relevant_columns"]
        indicators_apply = config.get("indicators_apply", None)
        target_column = config["target_column"]

        # trainer/modelo
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=(window_size, len(relevant_columns)), **config)
        model = trainer.load_model(model_path)

        processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)
        scaler_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "scaler")
        if norm_strategy == "global":
            processor.load_scaler(scaler_dir)

        # dados recentes
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

        if indicators_apply:
            df = TechnicalIndicators.process_indicators(df, indicators_apply)

        for c in relevant_columns + [target_column]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # janela de features (sem o alvo)
        feat_no_target = [c for c in relevant_columns if c != target_column]
        window_features = df[feat_no_target].values[-window_size:]          # (L, F)
        X_window = np.expand_dims(window_features, axis=0)                  # (1, L, F)

        if norm_strategy == "global":
            X_input, _ = processor.apply_normalization_global(X_window, np.zeros((1, steps_ahead)))
            y_pred_scaled = model.predict(X_input) if hasattr(model, "predict") else trainer.predict(model, X_input)
            y_pred = processor.inverse_transform_global(y_pred_scaled)

        elif norm_strategy == "local":
            target_idx = 0  # local (sem alvo no X) → y_mode que não exige alvo
            Xn, _y0, ctx = processor.apply_normalization_local(
                X_window, np.zeros((1, steps_ahead)),
                x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
            )
            y_pred_scaled = model.predict(Xn) if hasattr(model, "predict") else trainer.predict(model, Xn)
            y_pred = processor.inverse_transform_local(y_pred_scaled, ctx)

        else:  # "evomsn" ou "evomsn_like"
            Normalizer = EvoMSNNormalizer if norm_strategy == "evomsn" else EvoMSNLikeNormalizer
            # carrega meta salva no treino
            meta = Normalizer.load_meta(scaler_dir)
            if norm_strategy == "evomsn":
                msn = Normalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=len(feat_no_target),
                    target_idx=0,
                    k_scales=meta.k_scales,
                    agg=meta.agg,
                    random_state=config.get("seed", 42),
                    predictor_type=meta.predictor_type,
                )
            else:
                msn = Normalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=len(feat_no_target),
                    target_idx=0,
                    k_scales=meta.k_scales,
                    agg=meta.agg,
                    random_state=config.get("seed", 42),
                )
            msn.meta = meta
            # alvo por janela (para pesos FFT)
            target_win = df[target_column].values[-window_size:].reshape(1, window_size, 1)
            X_in, _y, ctx = msn.transform(X_window, None, target_windows=target_win)
            y_tilde = model.predict(X_in) if hasattr(model, "predict") else trainer.predict(model, X_in)
            _ps, y_pred = msn.denorm_and_ensemble(y_tilde, ctx)

        previsoes = [float(y_pred[0, i]) for i in range(steps_ahead)] if steps_ahead > 1 else [float(y_pred[0, 0])]

        return{
            "timestamp": str(df.iloc[-1]["timestamp"]) if "timestamp" in df.columns else str(datetime.utcnow()),
            "close_real": float(df.iloc[-1][target_column]),
            "previsoes": previsoes,
            "steps_ahead": steps_ahead,
            "target_column": target_column
        }
