# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/melb_data.csv
 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('DataSets\\melb_data.csv')
print(df, end='\n\n')

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])
    
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for colname in ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname']:
    df[colname] = le.fit_transform(df[colname])

print(df)

