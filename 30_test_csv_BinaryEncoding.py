# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import pandas as pd
from category_encoders.binary import BinaryEncoder

df = pd.read_csv('DataSets\\test.csv')

be = BinaryEncoder()
transformed_data = be.fit_transform(df[['RenkTercihi', 'Meslek']])
print(transformed_data)

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[transformed_data.columns] = transformed_data    # pd.concat((df, transformed_data), axis=1)
print(df)
