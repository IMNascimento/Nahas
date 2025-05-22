import itertools
import pandas as pd
from utils.validation import DataValidator 


class GridSearch:
    """
    Classe para realizar Grid Search em modelos de machine learning ou deep learning.
    """

    def __init__(self, model_trainer: callable, param_grid: dict, scoring: str, verbose: int = 1):
        """
        Inicializa a instância de GridSearch.

        :param model_trainer: Função ou callable que treina e avalia o modelo. Deve retornar a métrica definida em `scoring`.
        :param param_grid: Dicionário com os hiperparâmetros a serem otimizados e seus valores possíveis.
        :param scoring: Nome da métrica a ser usada como critério de avaliação (ex: 'accuracy', 'loss').
        :param verbose: Nível de logging (0 - silencioso, 1 - informações básicas, 2 - detalhado).
        """
        DataValidator.validate_dict(param_grid, key_type=str, value_type=list)
        DataValidator.validate_string(scoring)
        DataValidator.validate_integer(verbose, min_value=0, max_value=2)

        self._model_trainer = model_trainer
        self._param_grid = param_grid
        self._scoring = scoring
        self._verbose = verbose

    def _generate_configurations(self) -> list[dict]:
        """
        Gera todas as combinações possíveis de hiperparâmetros a partir de `_param_grid`.

        :return: Lista de dicionários com combinações de parâmetros.
        """
        # Filtrar apenas parâmetros que possuem valores
        filtered_param_grid = {k: v for k, v in self._param_grid.items() if v}

        if not filtered_param_grid:
            return []  # Retorna vazio se não houver parâmetros válidos

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
        """
        Realiza o Grid Search para encontrar os melhores hiperparâmetros.

        :param data_processor: Instância do DataProcessor para manipular os dados.
        :param data_df: DataFrame com os dados brutos.
        :param coluna_alvo: Nome da coluna alvo para previsão.
        :param steps_ahead: Número de passos à frente para previsão.
        :return: Dicionário com os melhores hiperparâmetros e resultados detalhados.
        """
        DataValidator.validate_string(coluna_alvo)
        DataValidator.validate_integer(steps_ahead, min_value=1)

        configurations = self._generate_configurations()
        best_score = -float("inf") if self._scoring != 'loss' else float("inf")
        best_params = None
        results = []

        if self._verbose > 0:
            print(f"Iniciando Grid Search com {len(configurations)} combinações...")

        for idx, config in enumerate(configurations):
            if self._verbose > 1:
                print(f"\n[{idx + 1}/{len(configurations)}] Testando configuração: {config}")

            try:
                # Atualizar janela de previsão dinamicamente
                window_size = config.get('window_size', 48)
                data_processor.window_size = window_size

                # Gerar janelas e dividir os dados
                X, y = data_processor.create_windows(data=data_df, coluna_alvo=coluna_alvo, steps_ahead=steps_ahead)
                X_train, X_validation, X_test, y_train, y_validation, y_test = data_processor.split_data(X, y)

                # Normalizar os dados
                X_train_scaled, y_train_scaled = data_processor.normalize(X_train, y_train)
                X_validation_scaled, y_validation_scaled = data_processor.apply_normalization(X_validation, y_validation)

                # Treina e avalia o modelo
                score = self._model_trainer(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled, **config)

                if self._verbose > 1:
                    print(f"Configuração: {config} | {self._scoring}: {score}")

                # Atualiza os melhores parâmetros se necessário
                if (self._scoring == 'loss' and score < best_score) or (self._scoring != 'loss' and score > best_score):
                    best_score = score
                    best_params = config

                # Salva o resultado
                results.append({**config, self._scoring: score})

            except Exception as e:
                if self._verbose > 0:
                    print(f"Erro ao testar configuração {config}: {e}")

        if self._verbose > 0:
            print("\nGrid Search concluído.")
            print(f"Melhores parâmetros: {best_params}")
            print(f"Melhor {self._scoring}: {best_score}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': pd.DataFrame(results)
        }
