import os
import pandas as pd
import numpy as np


class CSVExporter:
    """
    Classe responsável por salvar dados em formato CSV.
    Pode ser estendida para outros formatos (Parquet, Excel, etc.) se desejado.
    """
    def save_predictions_to_csv(
        self,
        timestamps,
        y_test,
        y_pred,
        output_path,
        steps_ahead=1,
        extra_columns=None,
    ):
        """
        Salva (timestamp, step, real_value, predicted_value) em formato longo.
        `extra_columns`: dict nome -> array (n_samples, n_steps) ou (n_samples,),
        gravado na mesma ordem (ex.: 'scale', a escala usada na inversao de y,
        que os intervalos conformais com escala local consomem).
        """
        timestamps = np.array(timestamps)
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        # Garante 2D
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        n_samples, n_steps = y_test.shape

        # Repete timestamp para cada step, empilha as previsões
        expanded_timestamps = np.repeat(timestamps, n_steps)
        steps = np.tile(np.arange(1, n_steps + 1), n_samples)
        results_df = pd.DataFrame({
            "timestamp": expanded_timestamps,
            "step": steps,
            "real_value": y_test.flatten(),
            "predicted_value": y_pred.flatten()
        })
        for nome, col in (extra_columns or {}).items():
            arr = np.asarray(col, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != n_samples:
                raise ValueError(f"extra_columns[{nome!r}] tem {arr.shape[0]} linhas; esperado {n_samples}.")
            if arr.shape[1] == 1 and n_steps > 1:
                arr = np.repeat(arr, n_steps, axis=1)
            results_df[nome] = arr[:, :n_steps].flatten()
        results_df.to_csv(output_path, index=False)
        print(f"Resultados salvos em: {output_path}")
        return results_df

    def save_generic_dataframe(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Salva qualquer DataFrame em CSV no caminho especificado.

        :param df: DataFrame a ser salvo.
        :param output_path: Caminho do arquivo CSV de saída.
        """
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        df.to_csv(output_path, index=False)
        print(f"DataFrame genérico salvo em: {output_path}")