# "Boston Housing Prices (BHP)"
# https://www.kaggle.com/datasets/vikrishnan/boston-house-prices

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt


df = pd.read_csv('DataSets\\housing.csv', delimiter=r'\s+', header=None)

# 8'inci sütun kategorik olduğu için o sütunu "one-hot-encoding" işlemine sokalım
highway_class = df.iloc[:, 8].to_numpy()

# OHE : One Hot Encoder işlemi yapıyoruz.
ohe = OneHotEncoder(sparse_output=False)
ohe_highway = ohe.fit_transform(highway_class.reshape(-1, 1)) #reshape ile 2 boyutlu (dataframe) hale getiriyoruz.

# OHE işleminden elde ettiğimiz matrisi DataFrame nesnesinin sonuna yerleştirebiliriz. 
# Tabii bundan önce bu sütunu silip "y" verilerini de ayırmalıyız:
dataset_y = df.iloc[:, -1].to_numpy()
df.drop([8, 13], axis=1, inplace=True) # 
dataset_x = pd.concat([df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()

# artık elimizde Numpy olarak dataset_x ve dataset_y verileri var.  
# Artık veri kümesini eğitim ve test olmak üzere ikiye ayırabiliriz:
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

# eğitim veri kümesini ölçekliyoruz
ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# Veri kümesi için iki saklı katman içeren klasik model kullanıyoruz
model = Sequential(name='Boston-Housing-Prices')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile('rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)
# Modelin çıktı katmanındaki aktivasyon fonksiyonunun "linear" olarak, 
# loss fonksiyonunun "mean_sequred_error", 
# metrik değerin de "mean_absolute_error" olarak kullandığımıza dikkat 

# çizimimizi yapalım.    
plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss']) # X : Epoch, Y: Loss
plt.plot(hist.epoch, hist.history['val_loss']) # X : Epoch, Y: val_Loss
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Mean Absolute Error - Validation Mean Absolute Error Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

"""
import pickle

model.save('boston-housing-prices.h5')
with open('boston-housing-prices.pickle', 'wb') as f:
    pickle.dump([ohe, ss], f)
"""

predict_df = pd.read_csv('DataSets\\predict-boston-hosing-prices.csv', delimiter=r'\s+', header=None)
"""
0.13554  12.50   6.070  0  0.4090  5.5940  36.80  6.4980   4  345.0  18.90 396.90  13.09  
0.12816  12.50   6.070  0  0.4090  5.8850  33.00  6.4980   4  345.0  18.90 396.90   8.79  
0.08826   0.00  10.810  0  0.4130  6.4170   6.60  5.2873   4  305.0  19.20 383.73   6.72  
0.15876   0.00  10.810  0  0.4130  5.9610  17.50  5.2873   4  305.0  19.20 376.94   9.88  
0.09164   0.00  10.810  0  0.4130  6.0650   7.80  5.2873   4  305.0  19.20 390.91   5.52  
0.19539   0.00  10.810  0  0.4130  6.2450   6.20  5.2873   4  305.0  19.20 377.17   7.54
"""

highway_class = predict_df.iloc[:, 8].to_numpy()
ohe_highway = ohe.transform(highway_class.reshape(-1, 1))
predict_df.drop(8, axis=1, inplace=True)
predict_dataset_x = pd.concat([predict_df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()
scaled_predict_dataset_x = ss.transform(predict_dataset_x )
predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    