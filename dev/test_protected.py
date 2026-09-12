import pandas as pd
import numpy as np
import datetime
import re

samples = pd.read_csv('dataset/sample_requests.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
events = pd.read_csv('dataset/financial_events.csv')
messages = pd.read_csv('dataset/messages.csv')
lookup = {
    'event_253': 4365000.0, 'event_1442': 100000.0, 'event_1545': 41272.0,
    'event_1700': 2854.0, 'event_1786': 704.05, 'event_3051': 1995.0,
    'event_3231': 8528.0, 'event_4535': 15339.0, 'event_5170': 723.0,
    'event_6033': 79679.26, 'event_6859': 3650.0, 'event_7307': 33.50,
    'event_7941': 2298.0, 'event_9421': 4543.0, 'event_9806': 9968.0,
    'event_10521': 393.22
}
for eid, amt in lookup.items():
    events.loc[events['event_id']==eid, 'amount'] = amt

for idx, r in samples.iterrows():
    u = r['user_id']
    req_d = datetime.datetime.strptime(r['request_date'], '%Y-%m-%d').date()
    p = profiles[profiles['user_id']==u].iloc[0]
    avail = p['current_available_balance']
    min_b = p['minimum_balance_to_keep']
    prot_cats = set([c.strip() for c in str(p['expense_categories_to_protect']).split('|') if c.strip()])
    
    # Calculate sum of protected expenses between req_d and next payday
    # Next payday
    sal_day = 15
    u_msgs = messages[messages['user_id']==u]
    no_sal = False
    for _, m in u_msgs.iterrows():
        if 'seasonal contract has ended' in m['message_text'].lower():
            no_sal = True
        m_date = re.search(r'(\d{4}-\d{2}-\d{2})', m['message_text'])
        if m_date and ('gaji' in m['message_text'].lower() or 'salary' in m['message_text'].lower() or 'payroll' in m['message_text'].lower()):
            sal_day = datetime.datetime.strptime(m_date.group(1), '%Y-%m-%d').date().day
            
    # events with "final" in description
    u_events = events[events['user_id']==u]
    if any('final' in str(d).lower() for d in u_events[u_events['category']=='salary']['description']):
        no_sal = True
        
    print(f"[{r['request_id']}] {u} | avail={avail} | min_b={min_b} | cushion={avail-min_b:.2f} | actual_safe={r['amount_safe_to_pay']} | diff={avail-min_b-r['amount_safe_to_pay']:.2f} | prot={prot_cats} | no_sal={no_sal}")
