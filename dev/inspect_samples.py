import sys
import datetime
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
import numpy as np

from main import FinancialDataReconciler, DATASET_DIR, CashFlowSimulator, PlanDecisionEngine

reconciler = FinancialDataReconciler(DATASET_DIR)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

def inspect_request(req_id):
    row = samples[samples['request_id'] == req_id].iloc[0]
    u_id = row['user_id']
    req_date = datetime.datetime.strptime(row['request_date'], '%Y-%m-%d').date()
    comp_date = datetime.datetime.strptime(row['desired_completion_date'], '%Y-%m-%d').date()
    req_amt = float(row['requested_amount'])
    
    user_facts = reconciler.get_user_facts(u_id, req_date)
    prof = user_facts['profile']
    
    print(f"=== {req_id} ({u_id}) ===")
    print(f"Req Date: {req_date}, Desired Date: {comp_date}, Req Amt: {req_amt}")
    print(f"Avail Bal: {prof['current_available_balance']}, Min Keep: {prof['minimum_balance_to_keep']}")
    print(f"Salary Amt: {user_facts['confirmed_salary_amt']}, Salary Day: {user_facts['confirmed_salary_day']}, Salary Ended: {user_facts['salary_ended']}")
    print("Recurring expenses:")
    for r in user_facts['recurring_expenses']:
        print(f"  {r['category']}: amt={r['amount']}, is_m={r['is_monthly']}, dom={r.get('day_of_month')}, flex={r['flexibility']}, eid={r['last_event_id']}")
    print("Pending debits:")
    for _, pd_row in user_facts['pending_debits'].iterrows():
        print(f"  {pd_row['event_id']}: amt={pd_row['amount']}, date={pd_row['settlement_date']}")
    
    print(f"Ground truth in sample_requests.csv:")
    print(f"  safe: {row['amount_safe_to_pay']}, status: {row['affordability_status']}, method: {row['recommended_payment_method']}")
    print(f"  plan: {row['payment_plan']}, earliest: {row['earliest_date_for_full_payment']}, changes: {row['spending_changes_needed']}")
    print()

for rid in ['request_06', 'request_08', 'request_11', 'request_12', 'request_13', 'request_21']:
    inspect_request(rid)
