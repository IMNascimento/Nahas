import os
from dotenv import load_dotenv

# Carregar o arquivo .env
load_dotenv()

class Settings:
    USER_DB = os.getenv("USER_DB")
    PASSWORD_DB = os.getenv("PASSWORD_DB")
    HOST_DB = os.getenv("HOST_DB")
    PORT_DB = os.getenv("PORT_DB")
    NAME_DB = os.getenv("NAME_DB")

    EMAIL_SMTP = os.getenv("EMAIL_SMTP")
    EMAIL_PORT = os.getenv("EMAIL_PORT")
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    EMAIL_IMAP = os.getenv("EMAIL_IMAP")

    USER_MT5 = os.getenv("LOGIN_MT5")
    PASSWORD_MT5 = os.getenv("PASSWORD_MT5")
    SERVER_MT5 = os.getenv("SERVER_MT5")

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET_KEY = os.getenv("BINANCE_API_SECRET_KEY")

    USE_GPU = os.getenv("USE_GPU")
    WINDOW_SIZE = os.getenv("WINDOW_SIZE")
    BATCH_SIZE = os.getenv("BATCH_SIZE")
    DROPOUT = os.getenv("DROPOUT")
    EPOCHS = os.getenv("EPOCHS")
    PATIENCE = os.getenv("PATIENCE")
    LAYERS_CONFIG = os.getenv("LAYERS_CONFIG")
    STEPS_AHEAD = os.getenv("STEPS_AHEAD")
    RELEVANT_COLUMNS = os.getenv("RELEVANT_COLUMNS")
    TARGET_COLUMN = os.getenv("TARGET_COLUMN")
    END_DATE = os.getenv("END_DATE")

    