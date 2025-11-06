# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/melb_data.csv


import pandas as pd

df = pd.read_csv('DataSets\\melb_data.csv')
#eksikveri = df.isna().sum(axis=0)

#Sütunlardaki eksik verilerin miktarları şöyle bulunabilir:
print(df.isna().sum()) #pd.isna(df).sum()

#Eksik verilerin toplam sayısı :
print(df.isna().sum().sum()) #pd.isna(df).sum().sum() 

#Eksik verilerin bulunduğu satır sayısı :
print(pd.isna(df).any(axis=1).sum())  #df.any().any(axis=1).sum() 

#Eksik verilerin bulunduğu satır indeksleri :
print(df.index[pd.isna(df).any(axis=1)]) #df.loc[df.isna().any(axis=1)].index        

missing_columns = [colname for colname in df.columns if df[colname].isna().any()]
print(f'Eksik verilerin bulunduğu sütunlar: {missing_columns}', end='\n\n')

missing_column_dist = df.isna().sum()
print('Eksik verilerin sütunlara göre dağılımı:')
print(missing_column_dist, end='\n\n')

missing_total = df.isna().sum().sum()
print(f'Eksik verilerin toplam sayısı: {missing_total}')

#df.shape #matrisin satır ve sütun sayısını verir.
#df.size #matrisin satırxsütun sayısını verir.
missing_ratio = missing_total / df.size
print(f'Eksik verilerin oranı: {missing_ratio}')

missing_rows = df.isna().any(axis=1).sum()
print(f'Eksik veri bulunan satırların sayısı: {missing_rows}')

missing_rows_ratio = missing_rows / len(df)
print(f'Eksik veri bulunan satırların oranı: {missing_rows_ratio}')
