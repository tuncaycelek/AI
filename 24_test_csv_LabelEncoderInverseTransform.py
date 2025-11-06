# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('DataSets\\test.csv')
print(df, end='\n\n')

le = LabelEncoder()

transformed_data = le.fit_transform(df['RenkTercihi'])
df['RenkTercihi'] = transformed_data

some_label_numbers = [0, 1, 1, 2, 2, 1]
label_names = le.inverse_transform(some_label_numbers)
print(label_names, end='\n\n')

transformed_data = le.fit_transform(df['Cinsiyet'])
df['Cinsiyet'] = transformed_data
print(df)

some_label_numbers = [0, 1, 1, 1, 0, 1]
label_names = le.inverse_transform(some_label_numbers)
print(label_names)