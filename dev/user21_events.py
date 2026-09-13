import sys
import datetime
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd

from main import FinancialDataReconciler, DATASET_DIR

reconciler = FinancialDataReconciler(DATASET_DIR)
events = reconciler.events[reconciler.events['user_id'] == 'user_21']
print(events[['event_id', 'event_date', 'category', 'amount', 'direction', 'flexibility', 'status', 'description']].to_string())
