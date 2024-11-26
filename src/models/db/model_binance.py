from peewee import Model, MySQLDatabase, FloatField, DateTimeField
from config.settings import Settings

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