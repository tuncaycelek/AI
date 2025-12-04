# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('DataSets\\test.csv')

le = LabelEncoder()

transformed_color = le.fit_transform(df['RenkTercihi'])
transformed_occupation = le.fit_transform(df['Meslek'])

ohe_color = to_categorical(transformed_color)
ohe_occupation = to_categorical(transformed_occupation)

color_categories = ['RenkTercihi_' + color for color in df['RenkTercihi'].unique()]
occupation_categories = ['Meslek_' + occupation for occupation in df['Meslek'].unique()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)

df[color_categories] = ohe_color
df[occupation_categories] = ohe_occupation

print(df)