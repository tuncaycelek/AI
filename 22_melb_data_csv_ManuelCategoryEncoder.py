# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/melb_data.csv
 
import pandas as pd

df = pd.read_csv('DataSets\\melb_data.csv')

def category_encoder(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
    
category_encoder(df, ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname'])
#print(df)
print(df['SellerG'])