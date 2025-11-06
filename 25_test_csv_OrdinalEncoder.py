# https://github.com/tuncaycelek/AI/blob/main/DataSets/test.csv
 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

    # Eğer kategorilere istediğiniz gibi değer vermek istiyorsanız bunu manuel bir biçimde yapabilirsiniz.
    # df = pd.read_csv('test.csv', converters={'EğitimDurumu': lambda s: {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}[s]})
    
df = pd.read_csv('DataSets\\test.csv')
print(df, end='\n\n')

oe = OrdinalEncoder()
transformed_data = oe.fit_transform(df[['Cinsiyet', 'RenkTercihi', 'EğitimDurumu']])
df[['Cinsiyet', 'RenkTercihi', 'EğitimDurumu']] = transformed_data

print(df, end='\n\n')
print(oe.categories_)