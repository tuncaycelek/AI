# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def ohe_encoder(s) : 
    le = LabelEncoder()
    transformed_data = le.fit_transform(s)
    
    um = np.eye(len(le.classes_))
    return um[transformed_data], le.classes_

df = pd.read_csv('DataSets\\test.csv')
#print(df, end='\n\n')

ohe_color, classes = ohe_encoder(df['RenkTercihi'])
df[classes] = ohe_color

df.drop(['RenkTercihi','Meslek'], axis=1, inplace=True)
print(df)