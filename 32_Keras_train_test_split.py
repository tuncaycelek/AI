# "Pima Indians Diabetes Database"
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataSets\\diabetes.csv')
#TRAINING_RATIO = 0.80
TEST_RATIO = 0.2

# (df == 0).sum()
si = SimpleImputer(strategy='mean', missing_values=0)

impute_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[impute_features] = si.fit_transform(df[impute_features])

dataset = df.to_numpy()

np.random.shuffle(dataset)

#dataset_x = dataset[:, :-1]
#dataset_y = dataset[:, -1]

#training_len = int(np.round(len(dataset_x) * TRAINING_RATIO))

#training_dataset_x = dataset_x[:training_len]
#test_dataset_x = dataset_x[training_len:]

#training_dataset_y = dataset_y[:training_len]
#test_dataset_y = dataset_y[training_len:]

dataset_x_df = df.iloc[:, :-1]
dataset_y_df = df.iloc[:, -1]

training_dataset_x_df, test_dataset_x_df, training_dataset_y_df, test_dataset_y_df = train_test_split(dataset_x_df, dataset_y_df, test_size=TEST_RATIO)

