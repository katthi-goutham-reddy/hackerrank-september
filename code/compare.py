import pandas as pd
import numpy as np
import datetime
import re

# Import the simulation from previous code and print per-sample results
from analyze import simulate_forecast, samples, profiles

for idx, r in samples.iterrows():
    p = profiles[profiles['user_id']==r['user_id']].iloc[0]
    curr = p['home_currency']
    pred = simulate_forecast(r['user_id'], r['request_date'], r['requested_amount'], amt_type='max')
    actual = r['amount_safe_to_pay']
    diff = pred - actual
    print(f"[{r['request_id']}] {curr:4s} | req={r['requested_amount']:10.2f} | actual_safe={actual:10.2f} | pred_safe={pred:10.2f} | diff={diff:10.2f}")
