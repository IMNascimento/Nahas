import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class Plotter:
    """
    Classe responsável por gerar plots de dados diversos,
    incluindo previsões e correlações.
    """

    def plot_price_predictions(
        self,
        results_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        real_col: str = "real_value",
        pred_col: str = "predicted_value",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plota os resultados previstos e reais e salva o gráfico se `save_path` for fornecido.

        :param results_df: DataFrame contendo as colunas de timestamp, valor real e valor previsto.
        :param timestamp_col: Nome da coluna de timestamp no DataFrame.
        :param real_col: Nome da coluna de valores reais no DataFrame.
        :param pred_col: Nome da coluna de valores previstos no DataFrame.
        :param save_path: Caminho do arquivo (PNG, etc.) a ser salvo. Se None, apenas exibe o gráfico.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(results_df[timestamp_col], results_df[real_col], label="Real")
        plt.plot(results_df[timestamp_col], results_df[pred_col], label="Previsto")
        plt.legend()
        plt.title("Previsão de Preço com LSTM")
        plt.xlabel("Timestamp")
        plt.ylabel("Preço")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plota uma matriz de correlação (heatmap) para as colunas selecionadas de um DataFrame.

        :param df: DataFrame contendo os dados.
        :param columns: Lista de colunas a serem incluídas na correlação. Se None, usa todas.
        :param save_path: Caminho do arquivo (PNG) a ser salvo. Se None, apenas exibe o gráfico.
        """
        if columns is not None:
            corr_df = df[columns].corr()
        else:
            corr_df = df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_df,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            annot_kws={"size": 6}  # Ajusta o tamanho da fonte dos números
        )
        plt.title("Matriz de Correlação")
        plt.tight_layout()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Matriz de correlação salva em: {save_path}")
        else:
            plt.show()

        plt.close()


    def plot_errors_over_time(
        self, y_test, y_pred, save_path: Optional[str] = None
    ) -> None:
        """
        Plota os erros absolutos ao longo do tempo.
        """
        errors = np.abs(y_test - y_pred)

        plt.figure(figsize=(10, 5))
        plt.plot(errors, label='Erro Absoluto Médio', color='red')
        plt.xlabel('Índice do Tempo')
        plt.ylabel('Erro Absoluto')
        plt.title('Erro Absoluto Médio ao Longo do Tempo')
        plt.legend()
        plt.grid()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Gráfico de erros salvo em: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_histogram_of_errors(
        self, y_test, y_pred, save_path: Optional[str] = None
    ) -> None:
        """
        Plota o histograma dos erros.
        """
        errors = y_test - y_pred

        plt.figure(figsize=(8, 5))
        plt.hist(errors, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Erro (y_real - y_previsto)')
        plt.ylabel('Frequência')
        plt.title('Histograma dos Erros')
        plt.grid()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Histograma de erros salvo em: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_scatter_real_vs_predicted(
        self, y_test, y_pred, save_path: Optional[str] = None
    ) -> None:
        """
        Plota um scatter plot de valores reais vs. previstos.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.title('Scatter Plot: Valores Reais vs. Previstos')
        plt.grid()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Scatter plot salvo em: {save_path}")
        else:
            plt.show()

        plt.close()