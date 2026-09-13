import sys
import datetime
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
from experiment_harness import CandidateSimulator, CandidateDecisionEngine, reconciler, samples

for rid in ['request_06', 'request_08', 'request_11', 'request_12', 'request_13', 'request_21']:
    row = samples[samples['request_id'] == rid].iloc[0]
    req_date = datetime.datetime.strptime(row['request_date'], '%Y-%m-%d').date()
    comp_date = datetime.datetime.strptime(row['desired_completion_date'], '%Y-%m-%d').date()
    req_amt = float(row['requested_amount'])
    user_facts = reconciler.get_user_facts(row['user_id'], req_date)
    
    print(f"\n==================== {rid} ====================")
    print(f"Ground truth: status={row['affordability_status']}, method={row['recommended_payment_method']}, earliest={row['earliest_date_for_full_payment']}, changes={row['spending_changes_needed']}")
    
    for cand in ['A', 'B', 'C', 'D']:
        engine = CandidateDecisionEngine(reconciler, candidate=cand)
        pred = engine.evaluate_request(row)
        print(f"Candidate {cand}: status={pred['affordability_status']}, method={pred['recommended_payment_method']}, safe={pred['amount_safe_to_pay']}, earliest={pred['earliest_date_for_full_payment']}, plan={pred['payment_plan']}, changes={pred['spending_changes_needed']}")
        
        # Also let's test if ground-truth spending change plan passes Candidate Simulator
        gt_changes = []
        if row['spending_changes_needed'] != 'none' and pd.notna(row['spending_changes_needed']):
            for part in str(row['spending_changes_needed']).split('|'):
                tokens = part.split(':')
                if tokens[0] == 'stop':
                    # Find category of event_id
                    ev = reconciler.events[reconciler.events['event_id'] == tokens[1]].iloc[0]
                    gt_changes.append({'action': 'stop', 'category': ev['category'], 'event_id': tokens[1], 'target_amt': 0.0})
                elif tokens[0] == 'reduce_to':
                    ev = reconciler.events[reconciler.events['event_id'] == tokens[1]].iloc[0]
                    gt_changes.append({'action': 'reduce_to', 'category': ev['category'], 'event_id': tokens[1], 'target_amt': float(tokens[2])})
            
            is_safe, min_cushion, balances = CandidateSimulator.simulate_plan(
                user_facts, req_date, comp_date,
                payment_schedule={req_date: req_amt},
                spending_changes=gt_changes,
                candidate=cand
            )
            print(f"  -> GT spending change is_safe({cand}) = {is_safe}, min_cushion = {min_cushion:.2f}")

