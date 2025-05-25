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
        """
        plt.figure(figsize=(10, 6))
        plt.plot(results_df[timestamp_col], results_df[real_col], label="Real")
        plt.plot(results_df[timestamp_col], results_df[pred_col], label="Previsto")
        plt.legend()
        plt.title("Previsão de Preço com LSTM/Transformer")
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

        if errors.ndim == 1 or errors.shape[1] == 1:
            plt.plot(errors, label='Erro Absoluto', color='red')
        else:
            for i in range(errors.shape[1]):
                plt.plot(errors[:, i], label=f'Erro Step {i+1}', alpha=0.7)
            plt.legend()

        plt.xlabel('Índice do Tempo')
        plt.ylabel('Erro Absoluto')
        plt.title('Erro Absoluto ao Longo do Tempo')
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
        Se multistep: plota todos juntos, mas cores diferentes, ou unifica.
        """
        errors = y_test - y_pred
        plt.figure(figsize=(8, 5))

        # Caso unificado: todos os erros num único histograma (mais simples de visualizar distribuição geral)
        if errors.ndim == 1 or errors.shape[1] == 1:
            plt.hist(errors.ravel(), bins=30, alpha=0.7, color='blue')
        else:
            # Plota um histograma para cada horizonte (step), sobrepondo.
            for i in range(errors.shape[1]):
                plt.hist(errors[:, i], bins=30, alpha=0.5, label=f"Step {i+1}", histtype='stepfilled')
            plt.legend()

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
        Se multistep, plota todos juntos (step1, step2, ...), mas pode separar se quiser.
        """
        plt.figure(figsize=(6, 6))

        if y_test.ndim == 1 or y_test.shape[1] == 1:
            plt.scatter(y_test.ravel(), y_pred.ravel(), alpha=0.5, color='purple', label="Step 1")
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, y_test.shape[1]))
            for i in range(y_test.shape[1]):
                plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, color=colors[i], label=f"Step {i+1}")

        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.title('Scatter Plot: Valores Reais vs. Previstos')
        plt.grid()
        plt.legend()

        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, format='png')
            print(f"Scatter plot salvo em: {save_path}")
        else:
            plt.show()

        plt.close()
