import sys
import datetime
import copy
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
import pandas as pd
import numpy as np

from main import FinancialDataReconciler, DATASET_DIR

reconciler = FinancialDataReconciler(DATASET_DIR)
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

class CandidateSimulator:
    @staticmethod
    def simulate_plan(user_facts: dict, request_date: datetime.date, desired_comp_date: datetime.date,
                      payment_schedule: dict = None, spending_changes: list = None,
                      candidate: str = 'A') -> tuple:
        p = user_facts['profile']
        avail_bal = float(p['current_available_balance'])
        min_bal_keep = float(p['minimum_balance_to_keep'])
        protect_cats = set([c.strip() for c in str(p['expense_categories_to_protect']).split('|') if c.strip()])

        if payment_schedule is None:
            payment_schedule = {}

        # Determine evaluation window and simulation start/end dates
        # Payment dates
        p_dates = []
        for p_d in payment_schedule.keys():
            if isinstance(p_d, str):
                p_dates.append(datetime.datetime.strptime(p_d, '%Y-%m-%d').date())
            else:
                p_dates.append(p_d)
        
        last_payment_date = max(p_dates) if p_dates else request_date
        first_payment_date = min(p_dates) if p_dates else request_date

        if candidate == 'A':
            # Fixed window [request_date, request_date + 90]
            sim_start = request_date
            sim_end = request_date + datetime.timedelta(days=90)
            eval_start = request_date
            eval_end = request_date + datetime.timedelta(days=90)
        elif candidate == 'B':
            # Window [request_date, max(desired_completion_date, last_payment_date)]
            sim_start = request_date
            eval_end = max(desired_comp_date, last_payment_date)
            sim_end = eval_end
            eval_start = request_date
        elif candidate == 'C':
            # Rolling 90 days from first (or last?) proposed payment date
            # Forecast runs for 90 days from payment date, but we must simulate from request_date
            sim_start = request_date
            sim_end = last_payment_date + datetime.timedelta(days=90)
            eval_start = request_date
            eval_end = last_payment_date + datetime.timedelta(days=90)
        elif candidate == 'D':
            # Fixed window [request_date, request_date + 90], but after deadline, only protected expenses count
            sim_start = request_date
            sim_end = request_date + datetime.timedelta(days=90)
            eval_start = request_date
            eval_end = request_date + datetime.timedelta(days=90)

        stopped_cats = set()
        reduced_cats = {}
        if spending_changes:
            for sc in spending_changes:
                if sc['action'] == 'stop':
                    stopped_cats.add(sc['category'])
                elif sc['action'] == 'reduce_to':
                    reduced_cats[sc['category']] = float(sc['target_amt'])

        num_days = (sim_end - sim_start).days + 1
        daily_deltas = {sim_start + datetime.timedelta(days=i): 0.0 for i in range(num_days)}

        # 1. Pending debits
        for _, pd_row in user_facts['pending_debits'].iterrows():
            s_date = datetime.datetime.strptime(pd_row['settlement_date'], '%Y-%m-%d').date()
            apply_date = max(request_date, s_date)
            if apply_date in daily_deltas:
                daily_deltas[apply_date] -= float(pd_row['amount'])

        # 2. Confirmed salary
        sal_amt = user_facts['confirmed_salary_amt']
        sal_day = user_facts['confirmed_salary_day']
        if sal_amt is not None and not user_facts['salary_ended']:
            # Up to 6 months
            for m_offset in range(6):
                year = request_date.year + (request_date.month - 1 + m_offset) // 12
                month = (request_date.month - 1 + m_offset) % 12 + 1
                try:
                    s_date = datetime.date(year, month, sal_day)
                except ValueError:
                    s_date = datetime.date(year, month, 28)
                if s_date in daily_deltas and s_date >= request_date:
                    daily_deltas[s_date] += sal_amt

        # 3. Recurring expenses
        for rec in user_facts['recurring_expenses']:
            cat = rec['category']
            if cat in stopped_cats:
                continue

            exp_amt = rec['amount']
            if cat in reduced_cats:
                exp_amt = reduced_cats[cat]

            if rec['is_monthly']:
                for m_offset in range(6):
                    year = request_date.year + (request_date.month - 1 + m_offset) // 12
                    month = (request_date.month - 1 + m_offset) % 12 + 1
                    try:
                        exp_date = datetime.date(year, month, rec['day_of_month'])
                    except ValueError:
                        exp_date = datetime.date(year, month, 28)
                    if exp_date in daily_deltas and exp_date > rec['last_date']:
                        if candidate == 'D' and exp_date > max(desired_comp_date, last_payment_date) and cat not in protect_cats:
                            pass # flexible expense after deadline not penalized in Candidate D
                        else:
                            daily_deltas[exp_date] -= exp_amt
            else:
                curr_date = rec['last_date'] + datetime.timedelta(days=int(rec['interval_days']))
                while curr_date <= sim_end:
                    if curr_date in daily_deltas and curr_date >= request_date:
                        if candidate == 'D' and curr_date > max(desired_comp_date, last_payment_date) and cat not in protect_cats:
                            pass
                        else:
                            daily_deltas[curr_date] -= exp_amt
                    curr_date += datetime.timedelta(days=int(rec['interval_days']))

        # 4. Proposed payment schedule
        for p_date, p_amt in payment_schedule.items():
            if isinstance(p_date, str):
                p_date = datetime.datetime.strptime(p_date, '%Y-%m-%d').date()
            if p_date in daily_deltas:
                daily_deltas[p_date] -= float(p_amt)

        # 5. Simulate day by day
        curr_bal = avail_bal
        min_cushion = float('inf')
        is_safe = True
        balances = {}

        for d in sorted(daily_deltas.keys()):
            curr_bal += daily_deltas[d]
            balances[d] = curr_bal
            if eval_start <= d <= eval_end:
                cushion = curr_bal - min_bal_keep
                if cushion < min_cushion:
                    min_cushion = cushion
                if curr_bal < min_bal_keep:
                    is_safe = False

        return is_safe, min_cushion, balances


class CandidateDecisionEngine:
    def __init__(self, reconciler: FinancialDataReconciler, candidate: str = 'A'):
        self.reconciler = reconciler
        self.candidate = candidate

    def evaluate_request(self, req_row: pd.Series) -> dict:
        req_id = req_row['request_id']
        user_id = req_row['user_id']
        req_date = datetime.datetime.strptime(req_row['request_date'], '%Y-%m-%d').date()
        desired_comp_date = datetime.datetime.strptime(req_row['desired_completion_date'], '%Y-%m-%d').date()
        requested_amt = float(req_row['requested_amount'])
        allows_partial = bool(req_row['allows_partial_payment'])

        user_facts = self.reconciler.get_user_facts(user_id, req_date)
        profile = user_facts['profile']
        home_curr = profile['home_currency']
        min_keep = float(profile['minimum_balance_to_keep'])

        considered_methods = [m.strip() for m in str(profile['payment_methods_user_will_consider']).split('|') if m.strip()]
        max_inst_months = float(profile['max_installment_months']) if pd.notna(profile['max_installment_months']) and str(profile['max_installment_months']).strip() != '' else 0.0

        # Step 1: Baseline simulation
        # For baseline cushion / amount_safe_to_pay today:
        # What is the evaluation window for baseline?
        # In candidate A: fixed 90 days from req_date
        # In candidate B: [req_date, desired_comp_date]
        # In candidate C: [req_date, req_date + 90]
        # In candidate D: [req_date, req_date + 90] with post-deadline flexible excluded
        _, baseline_cushion, _ = CandidateSimulator.simulate_plan(
            user_facts, req_date, desired_comp_date,
            payment_schedule={}, spending_changes=[], candidate=self.candidate
        )

        # Step 2: Compute amount_safe_to_pay today
        amount_safe_to_pay = max(0.0, min(requested_amt, baseline_cushion))
        if abs(amount_safe_to_pay - round(amount_safe_to_pay)) < 1e-4:
            amount_safe_to_pay = float(round(amount_safe_to_pay))
        else:
            amount_safe_to_pay = float(round(amount_safe_to_pay, 2))

        # Step 3: Compute earliest_date_for_full_payment
        earliest_full_date = None
        if amount_safe_to_pay >= requested_amt:
            earliest_full_date = req_date
        else:
            sal_day = user_facts['confirmed_salary_day']
            cand_dates = []
            if not user_facts['salary_ended']:
                for m_offset in range(4):
                    year = req_date.year + (req_date.month - 1 + m_offset) // 12
                    month = (req_date.month - 1 + m_offset) % 12 + 1
                    try:
                        p_date = datetime.date(year, month, sal_day)
                    except ValueError:
                        p_date = datetime.date(year, month, 28)
                    if p_date >= req_date and p_date <= req_date + datetime.timedelta(days=90):
                        cand_dates.append(p_date)

            for cand_d in sorted(cand_dates):
                is_safe, _, _ = CandidateSimulator.simulate_plan(
                    user_facts, req_date, desired_comp_date,
                    payment_schedule={cand_d: requested_amt}, candidate=self.candidate
                )
                if is_safe:
                    earliest_full_date = cand_d
                    break

        # Step 4: Enumerate candidate plans
        candidate_plans = []

        # (A) Full payment today without changes
        if 'full_payment' in considered_methods and amount_safe_to_pay >= requested_amt:
            candidate_plans.append({
                'status': 'affordable_now',
                'method': 'full_payment',
                'plan_str': f"{req_date.strftime('%Y-%m-%d')}:{self._fmt_amt(requested_amt)}",
                'first_date': req_date,
                'completion_date': req_date,
                'total_paid': requested_amt,
                'num_payments': 1,
                'changes_needed': 'none',
                'option_id': 'option_00'
            })

        # (B) Installments from request_payment_options.csv
        u_options = self.reconciler.payment_options[self.reconciler.payment_options['request_id'] == req_id]
        if 'installments' in considered_methods and max_inst_months > 0:
            for _, opt in u_options[u_options['payment_method'] == 'installments'].iterrows():
                num_p = int(opt['number_of_payments'])
                p_amt = float(opt['payment_amount'])
                f_date = datetime.datetime.strptime(opt['first_payment_date'], '%Y-%m-%d').date()
                freq_days = int(opt['payment_frequency_days'])
                tot_amt = float(opt['total_payable_amount'])
                opt_id = opt['payment_option_id']

                # Duration check
                duration_days = (num_p - 1) * freq_days
                duration_months = duration_days / 30.0
                if duration_months > max_inst_months + 0.1:
                    continue

                inst_sched = {}
                plan_parts = []
                for k in range(num_p):
                    inst_d = f_date + datetime.timedelta(days=k * freq_days)
                    inst_sched[inst_d] = p_amt
                    plan_parts.append(f"{inst_d.strftime('%Y-%m-%d')}:{self._fmt_amt(p_amt)}")

                last_pay_date = f_date + datetime.timedelta(days=(num_p - 1) * freq_days)

                is_safe, _, _ = CandidateSimulator.simulate_plan(
                    user_facts, req_date, desired_comp_date,
                    payment_schedule=inst_sched, candidate=self.candidate
                )
                if is_safe:
                    candidate_plans.append({
                        'status': 'affordable_with_plan',
                        'method': 'installments',
                        'plan_str': '|'.join(plan_parts),
                        'first_date': f_date,
                        'completion_date': last_pay_date,
                        'total_paid': tot_amt,
                        'num_payments': num_p,
                        'changes_needed': 'none',
                        'option_id': opt_id
                    })

        # (C) Partial payment
        if allows_partial and 'partial_payment' in considered_methods:
            if 0.0 < amount_safe_to_pay < requested_amt and earliest_full_date is not None:
                if earliest_full_date <= desired_comp_date:
                    remainder = requested_amt - amount_safe_to_pay
                    candidate_plans.append({
                        'status': 'affordable_with_plan',
                        'method': 'partial_payment',
                        'plan_str': f"{req_date.strftime('%Y-%m-%d')}:{self._fmt_amt(amount_safe_to_pay)}|{earliest_full_date.strftime('%Y-%m-%d')}:{self._fmt_amt(remainder)}",
                        'first_date': req_date,
                        'completion_date': earliest_full_date,
                        'total_paid': requested_amt,
                        'num_payments': 2,
                        'changes_needed': 'none',
                        'option_id': 'option_part'
                    })

        # (D) Wait
        if 'full_payment' in considered_methods and earliest_full_date is not None:
            if earliest_full_date > req_date and earliest_full_date <= desired_comp_date:
                candidate_plans.append({
                    'status': 'affordable_later',
                    'method': 'wait',
                    'plan_str': f"{earliest_full_date.strftime('%Y-%m-%d')}:{self._fmt_amt(requested_amt)}",
                    'first_date': earliest_full_date,
                    'completion_date': earliest_full_date,
                    'total_paid': requested_amt,
                    'num_payments': 1,
                    'changes_needed': 'none',
                    'option_id': 'option_wait'
                })

        # (E) Spending Changes
        if not any(p['method'] == 'full_payment' for p in candidate_plans) and 'full_payment' in considered_methods:
            spending_plan = self._find_spending_changes(user_facts, req_date, desired_comp_date, requested_amt, amount_safe_to_pay)
            if spending_plan:
                candidate_plans.append(spending_plan)

        # Rank plans
        best_plan = self._rank_plans(candidate_plans, desired_comp_date)

        if best_plan:
            affordability_status = best_plan['status']
            recommended_method = best_plan['method']
            payment_plan = best_plan['plan_str']
            spending_changes = best_plan['changes_needed']
        else:
            affordability_status = 'not_affordable'
            recommended_method = 'not_recommended'
            payment_plan = 'none'
            spending_changes = 'none'

        earliest_date_str = earliest_full_date.strftime('%Y-%m-%d') if earliest_full_date else ""
        if affordability_status == 'affordable_now':
            earliest_date_str = req_date.strftime('%Y-%m-%d')

        return {
            'request_id': req_id,
            'amount_safe_to_pay': amount_safe_to_pay,
            'affordability_status': affordability_status,
            'recommended_payment_method': recommended_method,
            'payment_plan': payment_plan,
            'earliest_date_for_full_payment': earliest_date_str,
            'spending_changes_needed': spending_changes,
        }

    def _find_spending_changes(self, user_facts: dict, req_date: datetime.date, desired_comp_date: datetime.date,
                               requested_amt: float, amount_safe_to_pay: float):
        profile = user_facts['profile']
        protect_cats = set([c.strip() for c in str(profile['expense_categories_to_protect']).split('|') if c.strip()])
        stop_cats = set([c.strip() for c in str(profile['expense_categories_user_is_willing_to_stop']).split('|') if c.strip()]) - protect_cats
        reduce_cats = set([c.strip() for c in str(profile['expense_categories_user_is_willing_to_reduce']).split('|') if c.strip()]) - protect_cats

        candidates = []
        for rec in user_facts['recurring_expenses']:
            cat = rec['category']
            flex = rec['flexibility']
            eid = rec['last_event_id']
            desc = rec['description']

            if cat in stop_cats and flex in ['stoppable', 'reducible_or_stoppable']:
                candidates.append({
                    'action': 'stop',
                    'category': cat,
                    'event_id': eid,
                    'desc': desc,
                    'target_amt': 0.0,
                    'savings': rec['amount']
                })
            elif cat in reduce_cats and flex in ['reducible', 'reducible_or_stoppable']:
                min_a = rec['minimum_allowed_amount']
                savings = rec['amount'] - min_a
                if savings > 0:
                    candidates.append({
                        'action': 'reduce_to',
                        'category': cat,
                        'event_id': eid,
                        'desc': desc,
                        'target_amt': min_a,
                        'savings': savings
                    })

        from itertools import combinations
        for k in range(1, min(4, len(candidates) + 1)):
            for comb in combinations(candidates, k):
                cats_in_comb = [c['category'] for c in comb]
                if len(cats_in_comb) != len(set(cats_in_comb)):
                    continue

                total_savings = sum(c['savings'] for c in comb)
                is_safe, _, _ = CandidateSimulator.simulate_plan(
                    user_facts,
                    req_date,
                    desired_comp_date,
                    payment_schedule={req_date: requested_amt},
                    spending_changes=list(comb),
                    candidate=self.candidate
                )

                if is_safe or (amount_safe_to_pay + total_savings >= requested_amt):
                    change_strs = []
                    for c in comb:
                        if c['action'] == 'stop':
                            change_strs.append(f"stop:{c['event_id']}")
                        else:
                            change_strs.append(f"reduce_to:{c['event_id']}:{self._fmt_amt(c['target_amt'])}")

                    return {
                        'status': 'affordable_with_plan',
                        'method': 'full_payment',
                        'plan_str': f"{req_date.strftime('%Y-%m-%d')}:{self._fmt_amt(requested_amt)}",
                        'first_date': req_date,
                        'completion_date': req_date,
                        'total_paid': requested_amt,
                        'num_payments': 1,
                        'changes_needed': '|'.join(change_strs),
                        'changes_list': list(comb),
                        'option_id': 'option_changes'
                    }

        return None

    def _rank_plans(self, plans: list, desired_comp_date: datetime.date):
        if not plans:
            return None
        on_time_plans = [p for p in plans if p['completion_date'] <= desired_comp_date]
        viable = on_time_plans if on_time_plans else []
        if not viable:
            return None

        def sort_key(p):
            has_changes = 0 if p['changes_needed'] == 'none' else 1
            tot_paid = p['total_paid']
            start_d = p['first_date']
            n_pay = p['num_payments']
            opt_id = p['option_id']
            return (has_changes, tot_paid, start_d, n_pay, opt_id)

        viable.sort(key=sort_key)
        return viable[0]

    def _fmt_amt(self, val: float) -> str:
        if abs(val - round(val)) < 1e-4:
            return f"{int(round(val))}"
        return f"{val:.2f}"

def run_all_candidates():
    candidates = ['A', 'B', 'C', 'D']
    results = {}
    
    for cand in candidates:
        engine = CandidateDecisionEngine(reconciler, candidate=cand)
        status_matches = 0
        method_matches = 0
        mismatches = []
        cand_preds = []
        
        for idx, row in samples.iterrows():
            pred = engine.evaluate_request(row)
            cand_preds.append(pred)
            s_match = (pred['affordability_status'] == row['affordability_status'])
            m_match = (pred['recommended_payment_method'] == row['recommended_payment_method'])
            if s_match:
                status_matches += 1
            if m_match:
                method_matches += 1
            if not (s_match and m_match):
                mismatches.append({
                    'req_id': row['request_id'],
                    'act_status': row['affordability_status'],
                    'pred_status': pred['affordability_status'],
                    'act_method': row['recommended_payment_method'],
                    'pred_method': pred['recommended_payment_method'],
                    'act_earliest': row['earliest_date_for_full_payment'],
                    'pred_earliest': pred['earliest_date_for_full_payment'],
                    'act_safe': row['amount_safe_to_pay'],
                    'pred_safe': pred['amount_safe_to_pay'],
                    'act_plan': row['payment_plan'],
                    'pred_plan': pred['payment_plan'],
                })
        results[cand] = {
            'status_matches': status_matches,
            'method_matches': method_matches,
            'mismatches': mismatches,
            'preds': cand_preds
        }

    return results

if __name__ == "__main__":
    results = run_all_candidates()
    print("=== SUMMARY TABLE ===")
    print(f"{'Candidate':<12} | {'Status Match':<14} | {'Method Match':<14}")
    print("-" * 50)
    for cand in ['A', 'B', 'C', 'D']:
        res = results[cand]
        print(f"Candidate {cand:<2} | {res['status_matches']}/25 ({res['status_matches']/25*100:.0f}%)    | {res['method_matches']}/25 ({res['method_matches']/25*100:.0f}%)")

    baseline_mismatches = {m['req_id'] for m in results['A']['mismatches']}
    print("\n=== FLIPS VS BASELINE (Candidate A) ===")
    for cand in ['B', 'C', 'D']:
        cand_mismatches = {m['req_id'] for m in results[cand]['mismatches']}
        flipped_to_right = baseline_mismatches - cand_mismatches
        flipped_to_wrong = cand_mismatches - baseline_mismatches
        print(f"\nCandidate {cand}:")
        print(f"  Flipped wrong -> right: {sorted(list(flipped_to_right)) if flipped_to_right else 'None'}")
        print(f"  Flipped right -> wrong: {sorted(list(flipped_to_wrong)) if flipped_to_wrong else 'None'}")
        print(f"  Remaining mismatches: {sorted(list(cand_mismatches)) if cand_mismatches else 'None'}")
        for m in results[cand]['mismatches']:
            print(f"    [{m['req_id']}] Act: {m['act_status']}/{m['act_method']} (safe={m['act_safe']}, earliest={m['act_earliest']}) vs Pred: {m['pred_status']}/{m['pred_method']} (safe={m['pred_safe']}, earliest={m['pred_earliest']})")

