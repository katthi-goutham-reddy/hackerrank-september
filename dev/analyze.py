import pandas as pd

rates = pd.read_csv('dataset/exchange_rates.csv')
print(rates.head(10))
print("Currencies in rates:", rates['from_currency'].unique(), "to", rates['to_currency'].unique())
