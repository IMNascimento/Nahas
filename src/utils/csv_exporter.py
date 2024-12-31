import os
import pandas as pd
from typing import Union, Optional, Any, List


class CSVExporter:
    """
    Classe responsável por salvar dados em formato CSV.
    Pode ser estendida para outros formatos (Parquet, Excel, etc.) se desejado.
    """
    def save_predictions_to_csv(
        self,
        timestamps: Union[List[Any], pd.Series, pd.Index],
        y_test: Union[List[float], pd.Series, pd.DataFrame],
        y_pred: Union[List[float], pd.Series, pd.DataFrame],
        output_path: str
    ) -> pd.DataFrame:
        """
        Salva os resultados (valores reais e previstos) em um arquivo CSV.

        :param timestamps: Coleção de timestamps (ou índices) correspondentes aos dados.
        :param y_test: Valores reais.
        :param y_pred: Valores previstos.
        :param output_path: Caminho do arquivo CSV de saída.
        :return: DataFrame com as colunas ["timestamp", "real_value", "predicted_value"].
        """
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        results_df = pd.DataFrame({
            "timestamp": timestamps,
            "real_value": pd.Series(y_test).values,
            "predicted_value": pd.Series(y_pred).values
        })
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