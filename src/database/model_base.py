from peewee import Model, MySQLDatabase
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