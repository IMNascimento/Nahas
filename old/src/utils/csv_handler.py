import os
import pandas as pd
from src.utils.logger import Logger

class CSVHandler:
    """
    Classe responsável por lidar com operações de leitura e escrita de arquivos CSV.
    """

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str, directory: str = "output"):
        """
        Salva o DataFrame em um arquivo CSV no diretório especificado. Se nenhum diretório for fornecido,
        os arquivos serão salvos na pasta 'output'.

        :param df: O DataFrame contendo os dados a serem salvos.
        :param filename: O nome do arquivo CSV onde os dados serão salvos.
        :param directory: O diretório onde o arquivo será salvo. Padrão é 'output'.
        """
        # Cria o diretório se ele não existir
        if not os.path.exists(directory):
            os.makedirs(directory)
            Logger.info(f"Directory '{directory}' created.")

        # Caminho completo para salvar o arquivo
        filepath = os.path.join(directory, filename)

        # Salva o arquivo CSV
        df.to_csv(filepath, index=False)
        Logger.info(f"Data saved to {filepath}")


    @staticmethod
    def read_from_csv(filename: str) -> pd.DataFrame:
        """
        Lê um arquivo CSV e retorna um DataFrame.

        :param filename: O nome do arquivo CSV a ser lido.
        :return: Um DataFrame contendo os dados do arquivo CSV.
        """
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            Logger.info(f"Data loaded from {filename}")
            return df
        else:
            Logger.error(f"File {filename} not found.")
            return pd.DataFrame()
    
    @staticmethod
    def read_multiple_csvs(file_list: list) -> pd.DataFrame:
        """
        Lê múltiplos arquivos CSV e retorna um DataFrame único contendo os dados concatenados.

        :param file_list: Lista de caminhos de arquivos CSV a serem lidos.
        :return: Um DataFrame concatenado contendo os dados de todos os arquivos CSV.
        """
        dataframes = []

        for file in file_list:
            if os.path.exists(file):
                df = pd.read_csv(file)
                dataframes.append(df)
                Logger.info(f"Data loaded from {file}")
            else:
                Logger.warning(f"File {file} not found.")

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            Logger.info("All CSV files concatenated into a single DataFrame.")
            return combined_df
        else:
            Logger.error("No files were loaded.")
            return pd.DataFrame()  # Retorna um DataFrame vazio se nenhum arquivo foi lido