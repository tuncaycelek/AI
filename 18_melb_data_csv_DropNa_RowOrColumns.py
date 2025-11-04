# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/diabetes.csv


import pandas as pd

df = pd.read_csv('DataSets\\melb_data.csv')

print(f'Veri kümesinin boyutu: {df.shape}')

df_deleted_rows = df.dropna(axis=0)
print(f'Satır atma sonucundaki yeni boyut: {df_deleted_rows.shape}')

df_deleted_cols = df.dropna(axis=1)
print(f'Sütun atma sonucundaki yeni boyut: {df_deleted_cols.shape}')