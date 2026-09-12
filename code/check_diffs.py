import pandas as pd

samples = pd.read_csv('dataset/sample_requests.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')

for idx, row in samples.iterrows():
    p = profiles[profiles['user_id'] == row['user_id']].iloc[0]
    avail = float(p['current_available_balance'])
    min_k = float(p['minimum_balance_to_keep'])
    diff = avail - min_k
    safe = float(row['amount_safe_to_pay'])
    req = float(row['requested_amount'])
    print(f"{row['request_id']} ({row['user_id']}): avail={avail:.2f}, min_k={min_k:.2f}, diff={diff:.2f}, safe={safe:.2f}, req={req:.2f}, safe_ratio={safe/diff if diff>0 else 0:.3f}")
