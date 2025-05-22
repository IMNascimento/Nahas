from peewee import Model, MySQLDatabase, FloatField, DateTimeField
from config.settings import Settings
import pandas as pd

# Configuração do banco de dados MySQL
db = MySQLDatabase(
    Settings.NAME_DB,
    user=Settings.USER_DB,
    password=Settings.PASSWORD_DB,
    host=Settings.HOST_DB,
    port=int(Settings.PORT_DB)
)

class BaseModel(Model):
    class Meta:
        database = db

# Defina o modelo para cotações horárias
class HourlyQuote(BaseModel):
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
        query = (HourlyQuote
                .select()
                .where(HourlyQuote.timestamp <= end_date)
                .order_by(HourlyQuote.timestamp))
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
        query = (HourlyQuote
                .select()
                .where(HourlyQuote.timestamp >= start_date)
                .order_by(HourlyQuote.timestamp))
        
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
        query = (HourlyQuote
                .select()
                .where((HourlyQuote.timestamp >= start_date) & (HourlyQuote.timestamp <= end_date))
                .order_by(HourlyQuote.timestamp))

        data = pd.DataFrame(list(query.dicts()))
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data[["timestamp","open", "high", "low", "close", "volume"]]
        else:
            print("Nenhum dado encontrado no banco para o período especificado.")
            return None



# Defina o modelo para cotações diárias
class DailyQuote(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

# Defina o modelo para cotações semanais
class WeeklyQuote(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

class MonthlyQuote(BaseModel):
    timestamp = DateTimeField(unique=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

# Crie as tabelas no banco de dados
db.connect()
db.create_tables([HourlyQuote, DailyQuote, WeeklyQuote, MonthlyQuote])
db.close()




