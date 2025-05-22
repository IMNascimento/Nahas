import itertools
import pandas as pd
from utils.validation import DataValidator
from datetime import datetime
import json
import uuid
from database.model_nahas import Experiment, TrainingRun, GridResult  

class GridSearch:
    """
    Classe para realizar Grid Search em modelos de machine learning ou deep learning, agora com persistência.
    """

    def __init__(self, model_trainer: callable, param_grid: dict, scoring: str, experiment_meta: dict = None, verbose: int = 1):
        """
        :param model_trainer: Função/callable que treina e avalia o modelo.
        :param param_grid: Dicionário de hiperparâmetros e valores possíveis.
        :param scoring: Métrica de avaliação ('loss', 'accuracy' etc).
        :param experiment_meta: Dict extra com meta-infos (type, framework, model_type, user etc).
        :param verbose: Logging.
        """
        DataValidator.validate_dict(param_grid, key_type=str, value_type=list)
        DataValidator.validate_string(scoring)
        DataValidator.validate_integer(verbose, min_value=0, max_value=2)

        self._model_trainer = model_trainer
        self._param_grid = param_grid
        self._scoring = scoring
        self._verbose = verbose
        self._experiment_meta = experiment_meta or {}

    def _generate_configurations(self) -> list[dict]:
        filtered_param_grid = {k: v for k, v in self._param_grid.items() if v}
        if not filtered_param_grid:
            return []
        keys = filtered_param_grid.keys()
        values = filtered_param_grid.values()
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    def search(
        self,
        data_processor,
        data_df: pd.DataFrame,
        coluna_alvo: str,
        steps_ahead: int
    ) -> dict:
        DataValidator.validate_string(coluna_alvo)
        DataValidator.validate_integer(steps_ahead, min_value=1)

        configurations = self._generate_configurations()
        best_score = -float("inf") if self._scoring != 'loss' else float("inf")
        best_params = None
        results = []

        # === 1. Salva experimento no banco ===
        experiment = Experiment.create(
            type=self._experiment_meta.get("type", "grid"),
            framework=self._experiment_meta.get("framework", "keras"),
            model_type=self._experiment_meta.get("model_type", "lstm"),
            hyperparams_json=json.dumps(self._param_grid),
            hash_config=self._experiment_meta.get("hash_config", str(uuid.uuid4())),
            user=self._experiment_meta.get("user"),
            status="running",
            notes=self._experiment_meta.get("notes"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # === 2. Salva o run principal do grid (um run para todo grid) ===
        training_run = TrainingRun.create(
            experiment=experiment,
            run_uuid=str(uuid.uuid4()),
            start_time=datetime.now(),
            status="running",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        if self._verbose > 0:
            print(f"Iniciando Grid Search com {len(configurations)} combinações...")

        for idx, config in enumerate(configurations):
            if self._verbose > 1:
                print(f"\n[{idx + 1}/{len(configurations)}] Testando configuração: {config}")

            try:
                window_size = config.get('window_size', 48)
                data_processor.window_size = window_size

                X, y = data_processor.create_windows(data=data_df, coluna_alvo=coluna_alvo, steps_ahead=steps_ahead)
                X_train, X_validation, X_test, y_train, y_validation, y_test = data_processor.split_data(X, y)

                X_train_scaled, y_train_scaled = data_processor.normalize(X_train, y_train)
                X_validation_scaled, y_validation_scaled = data_processor.apply_normalization(X_validation, y_validation)

                # ===== Chama treino e avaliação do modelo =====
                metricas = self._model_trainer(
                    X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled, **config
                )
                # metricas pode ser um dict: {"val_loss": ..., "train_loss": ..., ...}

                score = metricas[self._scoring] if isinstance(metricas, dict) else metricas

                if self._verbose > 1:
                    print(f"Configuração: {config} | {self._scoring}: {score}")

                if (self._scoring == 'loss' and score < best_score) or (self._scoring != 'loss' and score > best_score):
                    best_score = score
                    best_params = config

                # ===== Salva resultado do grid no banco =====
                grid_result = GridResult.create(
                    training_run=training_run,
                    params_json=json.dumps(config),
                    train_loss=metricas.get("train_loss"),
                    val_loss=metricas.get("val_loss"),
                    metrics_json=json.dumps(metricas),
                    model_path=metricas.get("model_path"),         # Passe o caminho salvo ao treinar
                    csv_metrics_path=metricas.get("csv_metrics_path"),  # Idem
                    epoch=metricas.get("best_epoch"),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                results.append({**config, **metricas})

            except Exception as e:
                if self._verbose > 0:
                    print(f"Erro ao testar configuração {config}: {e}")

        # Finaliza status do experimento/run
        training_run.status = "finished"
        training_run.end_time = datetime.now()
        training_run.save()
        experiment.status = "finished"
        experiment.updated_at = datetime.now()
        experiment.save()

        if self._verbose > 0:
            print("\nGrid Search concluído.")
            print(f"Melhores parâmetros: {best_params}")
            print(f"Melhor {self._scoring}: {best_score}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': pd.DataFrame(results),
            'experiment_id': experiment.id,
            'training_run_id': training_run.id
        }