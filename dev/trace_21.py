import sys
import datetime
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
import numpy as np

from main import FinancialDataReconciler, DATASET_DIR, CashFlowSimulator

reconciler = FinancialDataReconciler(DATASET_DIR)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

row = samples[samples['request_id'] == 'request_21'].iloc[0]
req_date = datetime.datetime.strptime(row['request_date'], '%Y-%m-%d').date()
user_facts = reconciler.get_user_facts('user_21', req_date)

is_safe, min_cushion, balances = CashFlowSimulator.simulate(user_facts, req_date)
print("request_21 baseline min_cushion:", min_cushion)
for d, bal in sorted(balances.items())[:20]:
    print(f"{d}: bal={bal:.2f}, cushion={bal - float(user_facts['profile']['minimum_balance_to_keep']):.2f}")
