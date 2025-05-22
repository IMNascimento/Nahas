from peewee import FloatField, DateTimeField
import pandas as pd
from database.model_base import BaseModel, db

# Defina o modelo para cotações horárias
class HourlyQuoteBitcoin(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    def get_to_date(end_date):
        """
        Carrega os dados até uma data específica do banco de dados.
        """
        query = (HourlyQuoteBitcoin
                .select()
                .where(HourlyQuoteBitcoin.timestamp <= end_date)
                .order_by(HourlyQuoteBitcoin.timestamp))
        data = pd.DataFrame(list(query.dicts()))
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        else:
            print("Nenhum dado encontrado no banco.")
            return pd.DataFrame()
    
    def get_from_date(start_date):
        """
        Carrega os dados a partir de uma data específica do banco de dados.
        """
        query = (HourlyQuoteBitcoin
                .select()
                .where(HourlyQuoteBitcoin.timestamp >= start_date)
                .order_by(HourlyQuoteBitcoin.timestamp))
        
        data = pd.DataFrame(list(query.dicts()))
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data[["open", "high", "low", "close", "volume"]]
        else:
            print("Nenhum dado encontrado no banco.")
            return None

    def get_between_dates(start_date, end_date):
        """
        Carrega os dados de cotações entre uma data inicial e final.

        :param start_date: Data inicial no formato 'YYYY-MM-DD HH:MM:SS'.
        :param end_date: Data final no formato 'YYYY-MM-DD HH:MM:SS'.
        :return: DataFrame com os dados filtrados.
        """
        query = (HourlyQuoteBitcoin
                .select()
                .where((HourlyQuoteBitcoin.timestamp >= start_date) & (HourlyQuoteBitcoin.timestamp <= end_date))
                .order_by(HourlyQuoteBitcoin.timestamp))

        data = pd.DataFrame(list(query.dicts()))
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data[["timestamp","open", "high", "low", "close", "volume"]]
        else:
            print("Nenhum dado encontrado no banco para o período especificado.")
            return None



# Defina o modelo para cotações diárias
class DailyQuoteBitcoin(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

# Defina o modelo para cotações semanais
class WeeklyQuoteBitcoin(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

class MonthlyQuoteBitcoin(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

# Crie as tabelas no banco de dados
db.connect()
db.create_tables([HourlyQuoteBitcoin, DailyQuoteBitcoin, WeeklyQuoteBitcoin, MonthlyQuoteBitcoin], safe=True)
db.close()




