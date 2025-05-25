from .model_base import db
from .model_binance import HourlyQuoteBitcoin, DailyQuoteBitcoin, WeeklyQuoteBitcoin, MonthlyQuoteBitcoin
from .model_nahas import TrainingRun, FineTuningRun, GridResult


def init_db():
    if db.is_closed():
        db.connect()
    db.create_tables([HourlyQuoteBitcoin, DailyQuoteBitcoin, WeeklyQuoteBitcoin, MonthlyQuoteBitcoin, TrainingRun, FineTuningRun, GridResult], safe=True)
    db.close()
