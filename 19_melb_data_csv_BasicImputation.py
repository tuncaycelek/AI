# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/diabetes.csv


import pandas as pd

df = pd.read_csv('DataSets\\melb_data.csv')

missing_total = df.isna().sum().sum()
print(f'Eksik verilerin toplam sayısı: {missing_total}')

impute_val = df['Car'].mean().round()
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = df['BuildingArea'].mean().round()
df['BuildingArea'] = df['BuildingArea'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = df['YearBuilt'].median()
df['YearBuilt'] = df['YearBuilt'].fillna(impute_val)    # eşdeğeri df['YearBuilt'].fillna(impute_val, inplace=True)

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])    # eşdeğeri df['CouncilArea'].fillna(impute_val, inplace=True)

missing_total = df.isna().sum().sum()
print(f'Imputation sonrası Eksik verilerin toplam sayısı: {missing_total}')
