import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
for uid, target in [('user_08', 452.0), ('user_14', 1134.0), ('user_18', 624.0), ('user_21', 568.0), ('user_22', 157.0)]:
    u_ev = events[events['user_id'] == uid]
    print(f"=== {uid} target={target} ===")
    debits = u_ev[u_ev['direction'] == 'debit']
    print(debits[['category', 'amount', 'event_date', 'status', 'flexibility']].to_string())
