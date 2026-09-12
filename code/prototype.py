import pandas as pd
import numpy as np
import datetime

events = pd.read_csv('dataset/financial_events.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
samples = pd.read_csv('dataset/sample_requests.csv')
messages = pd.read_csv('dataset/messages.csv')

# Test recurrence detection on sample users
def get_recurring_series(user_events, req_date):
    # Only settled or scheduled events up to req_date
    ev = user_events[user_events['direction'] == 'debit'].copy()
    ev = ev[ev['status'].isin(['settled', 'scheduled', 'pending'])]
    
    series_list = []
    for cat, grp in ev.groupby('category'):
        grp = grp.sort_values('event_date')
        # Check if recurring
        if len(grp) >= 2:
            dates = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in grp['event_date']]
            diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            median_diff = np.median(diffs)
            last_event = grp.iloc[-1]
            last_amt = last_event['amount']
            max_amt = grp['amount'].max()
            mean_amt = grp['amount'].mean()
            
            freq = 'unknown'
            if 25 <= median_diff <= 35:
                freq = 'monthly'
            elif 5 <= median_diff <= 9:
                freq = 'weekly'
            elif 12 <= median_diff <= 16:
                freq = 'biweekly'
            elif 18 <= median_diff <= 23:
                freq = 'triweekly'
            
            series_list.append({
                'category': cat,
                'freq': freq,
                'median_diff': median_diff,
                'last_date': dates[-1],
                'day_of_month': dates[-1].day,
                'last_amt': last_amt,
                'max_amt': max_amt,
                'mean_amt': mean_amt,
                'flexibility': last_event['flexibility'],
                'min_allowed': last_event['minimum_allowed_amount'],
                'last_event_id': last_event['event_id'],
                'description': last_event['description']
            })
    return series_list

for u in ['user_01', 'user_02', 'user_03', 'user_08', 'user_15', 'user_18']:
    s = get_recurring_series(events[events['user_id']==u], '2024-01-01')
    print(f"=== {u} ===")
    for item in s:
        print(f"  {item['category']:20s}: {item['freq']:8s} (diff={item['median_diff']:.1f}d), last={item['last_amt']}, max={item['max_amt']}, day={item['day_of_month']}")
