from .model_base import db
#from .model_binance import HourlyQuoteBitcoin, DailyQuoteBitcoin, WeeklyQuoteBitcoin, MonthlyQuoteBitcoin
from .model_nahas import TrainingRun, FineTuningRun, GridResult
from .model_nocapital import PriceHistory


def init_db():
    if db.is_closed():
        db.connect()
    db.create_tables([ TrainingRun, FineTuningRun, GridResult, PriceHistory], safe=True)
    db.close()
