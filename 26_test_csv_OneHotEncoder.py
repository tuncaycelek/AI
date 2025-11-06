# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv
 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('DataSets\\test.csv')

    #one-hot-encoding uygulamanın diğer bir yolu Pandas kütüphanesindeki get_dummies fonksiyonunu kullanmaktır
#print(df)
#transformed_df = pd.get_dummies(df, dtype='uint8')
#transformed_df = pd.get_dummies(df.iloc[:, 1:], dtype='uint8')
#print(transformed_df)

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')
transformed_data = ohe.fit_transform(df[['RenkTercihi', 'Meslek']])
df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)

categories1 = ['RenkTercihi_' + category for category in ohe.categories_[0]]
categories2 = ['Meslek_' + category for category in ohe.categories_[1]]

df[categories1 + categories2] = transformed_data

print(df)

