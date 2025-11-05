# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/diabetes.csv

    #MHS veri kümesi üzerinde eksik verilerin bulunduğu sütunlar sckit-learn SimpleImputer sınıfı kullanılarak 
    #doldurulmuştur.
    
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('DataSets\\melb_data.csv')

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])

missing_column_dist = df.isna().sum()
print('Eksik verilerin sütunlara göre dağılımı:')
print(missing_column_dist, end='\n\n')