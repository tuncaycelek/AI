# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('DataSets\\test.csv')
#print(df, end='\n\n')

color_cats = np.unique(df['RenkTercihi'].to_numpy())
occupation_cats = np.unique(df['Meslek'].to_numpy())

le = LabelEncoder()
df['RenkTercihi'] = le.fit_transform(df['RenkTercihi'])
df['Meslek'] = le.fit_transform(df['Meslek'])

#print(df, end='\n\n')

    #NumPy'ın eye fonksiyonundan faydalanmaktır. Bu fonksiyon bize birim matris verir
color_um = np.eye(len(color_cats))
occupation_um = np.eye(len(occupation_cats))
#print(color_um, end='\n\n')
#print(occupation_um, end='\n\n')

    # Bir NumPy     dizisi bir listeyle indekslenebildiğine göre bu birim matris LabelEncoder 
    # ile sayısal biçime dönüştürülmüş bir dizi ile indekslenirse istenilen dönüştürme yapılmış olur.    
ohe_color = color_um[df['RenkTercihi'].to_numpy()]
ohe_occupation = occupation_um[df['Meslek'].to_numpy()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[color_cats] = ohe_color
df[occupation_cats] = ohe_occupation

print(df, end='\n\n')