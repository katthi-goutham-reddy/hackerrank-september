import pandas as pd
from main import FinancialDataReconciler, PlanDecisionEngine, DATASET_DIR

reconciler = FinancialDataReconciler(DATASET_DIR)
engine = PlanDecisionEngine(reconciler)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

for idx, row in samples.iterrows():
    pred = engine.evaluate_request(row)
    act_status = row['affordability_status']
    pred_status = pred['affordability_status']
    act_method = row['recommended_payment_method']
    pred_method = pred['recommended_payment_method']
    
    if act_status != pred_status or act_method != pred_method:
        print(f"[{row['request_id']}] MISMATCH:")
        print(f"   Actual: status={act_status}, method={act_method}, safe={row['amount_safe_to_pay']}, earliest={row['earliest_date_for_full_payment']}")
        print(f"   Pred:   status={pred_status}, method={pred_method}, safe={pred['amount_safe_to_pay']}, earliest={pred['earliest_date_for_full_payment']}")
        print(f"   Plan Act:  {row['payment_plan']}")
        print(f"   Plan Pred: {pred['payment_plan']}")
        print(f"   Expl Act:  {row['decision_explanation']}")
        print(f"   Expl Pred: {pred['decision_explanation']}")
        print()
