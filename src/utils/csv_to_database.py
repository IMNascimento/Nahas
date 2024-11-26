import pandas as pd
from peewee import Model
from typing import List
import os

class CSVToDatabase:
    """
    Classe para processar arquivos CSV e inserir os dados no banco de dados.
    """

    def __init__(self, model: Model, required_columns: List[str]):
        """
        Inicializa a classe com o modelo Peewee e as colunas obrigatórias.

        :param model: Modelo Peewee onde os dados serão inseridos.
        :param required_columns: Lista de colunas obrigatórias para validação.
        """
        self.model = model
        self.required_columns = required_columns

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Lê um arquivo CSV e retorna um DataFrame do Pandas.

        :param file_path: Caminho para o arquivo CSV.
        :return: DataFrame contendo os dados do CSV.
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Erro ao ler o arquivo CSV {file_path}: {e}")

    def validate_columns(self, df: pd.DataFrame):
        """
        Valida se as colunas do DataFrame correspondem às exigidas.

        :param df: DataFrame para validação.
        :raise ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"As colunas obrigatórias estão ausentes: {missing_columns}")

    def insert_data(self, df: pd.DataFrame):
        """
        Insere os dados do DataFrame no banco de dados.

        :param df: DataFrame contendo os dados a serem inseridos.
        """
        try:
            with self.model._meta.database.atomic():  # Inserção em lotes
                batch_size = 100
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i + batch_size].to_dict(orient='records')
                    self.model.insert_many(batch).execute()
        except Exception as e:
            raise ValueError(f"Erro ao inserir os dados no banco: {e}")

    def process_files(self, file_paths: List[str]):
        """
        Processa uma lista de arquivos CSV e insere os dados no banco.

        :param file_paths: Lista de caminhos para os arquivos CSV.
        """
        for file_path in file_paths:
            print(f"Processando arquivo: {file_path}")
            try:
                # Lê o arquivo CSV
                df = self.read_csv(file_path)

                # Valida as colunas
                self.validate_columns(df)

                # Insere os dados no banco
                self.insert_data(df)

                print(f"Arquivo {file_path} processado com sucesso!")
            except Exception as e:
                print(f"Erro ao processar o arquivo {file_path}: {e}")










"""
from models.db.model_binance import HourlyQuote, DailyQuote, WeeklyQuote, MonthlyQuote
from utils.csv_to_database import CSVToDatabase

# Defina os arquivos CSV
csv_files = [
    "dani/BTCUSDT_week_2013_2013.csv",
    "dani/BTCUSDT_week_2014_2014.csv",
    "dani/BTCUSDT_week_2015_2015.csv",
    "dani/BTCUSDT_week_2016_2016.csv",
    "dani/BTCUSDT_week_2017_2017.csv",
    "dani/BTCUSDT_week_2018_2018.csv",
    "dani/BTCUSDT_week_2019_2019.csv",
    "dani/BTCUSDT_week_2020_2020.csv",
    "dani/BTCUSDT_week_2021_2021.csv",
    "dani/BTCUSDT_week_2022_2022.csv",
    "dani/BTCUSDT_week_2023_2023.csv",
    "dani/BTCUSDT_week_2024_2024.csv",
    ]

# Colunas obrigatórias
required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

# Inicializa a classe com o modelo e as colunas
csv_processor = CSVToDatabase(model=WeeklyQuote, required_columns=required_columns)

# Processa os arquivos
csv_processor.process_files(csv_files)
"""