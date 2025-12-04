# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('DataSets\\test.csv')

    # OneHotEncoder sınıfının drop parametresi 'first' olarak geçilirse bu durumda transform işlemi 
    # "dummy variable encoding" biçiminde yapılmaktadır
ohe = OneHotEncoder(sparse_output=False, drop='first')
transformed_data = ohe.fit_transform(df[['RenkTercihi']])

    # Pandas'ın get_dummies fonksiyonunda drop_first parametresi True geçilirse "dummy variable encoding" uygulanmaktadır. 
#transformed_df = pd.get_dummies(df, columns=['RenkTercihi', 'Meslek'], dtype='uint8', drop_first=True)

print(df['RenkTercihi'])
print(ohe.categories_)
print(transformed_data)
