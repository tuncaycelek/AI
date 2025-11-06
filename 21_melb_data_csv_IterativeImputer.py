# "Melbourne Housing Snapshot (MHS)"
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot?resource=download
# https://github.com/tuncaycelek/AI/blob/main/DataSets/diabetes.csv

    #üç sütuna dayalı olarak tahminleme yöntemiyle doldurmaya örnek verilmiştir. 
    #Detay : https://scikit-learn.org/1.5/modules/generated/sklearn.impute.IterativeImputer.html
    
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('DataSets\\melb_data.csv')


si = IterativeImputer()
df[['Car', 'BuildingArea', 'YearBuilt']] = si.fit_transform(df[['Car', 'BuildingArea', 'YearBuilt']])