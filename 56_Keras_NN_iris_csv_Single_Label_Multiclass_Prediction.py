# "Iris Species"
# https://www.kaggle.com/datasets/uciml/iris?resource=download

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt


df = pd.read_csv('DataSets\\Iris.csv')

# ilk sütunu ve son sütunu atıyoruz.
dataset_x = df.iloc[:, 1:-1].to_numpy(dtype='float32')

# OHE işlemini yapıyoruz. (son sütun : Iris-setosa, Iris-virginica..)
ohe = OneHotEncoder(sparse= False)
dataset_y = ohe.fit_transform(df.iloc[:, -1].to_numpy().reshape(-1, 1))
# ohe.categories_ # : sıraya dizildiğine dikkat

# veri setlerimizi ayırıyoruz x-y, training-test
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

# Standard Scaling işlemimizi yapıyoruz.
ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)

# modelimizi oluşturalım.
# çıktı katmanındaki aktivasyon fonksiyonu "softmax" olmalıdır. 
model = Sequential(name='Iris')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(dataset_y.shape[1], activation='softmax', name='Output')) 
# Dense(dataset_y.shape[1] = 3 yani çıktı katmanında 3 sınıf var. (Iris-virgin, Iris-.., ...)
model.summary()

# çok sınıflı sınıflandırma problemlerindeki loss fonksiyonu "categorical_crossentropy", 
# Metrik değer olarak "binary_accuracy" yerine "categorical_accuracy" kullanılmalıdır.
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model kuruldu. eğitimi 100 epoch olarak başlatalım. 
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

# çizimlerimizi yapalım.
plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Categorical Accuracy - Validation Categorical Accuracy', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

# scale edilmiş test dataset'imiz üzerinden testimizi yapıyoruz.
scaled_test_dataset_x = ss.transform(test_dataset_x)
eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
#test sonuçları
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

#tahminleme:
predict_dataset_x = pd.read_csv('DataSets\\predict-iris.csv').to_numpy(dtype='float32') #tahminleyeceğimiz dataset
scaled_predict_dataset_x = ss.transform(predict_dataset_x) # scale ettik.

# işlem :
predict_result = model.predict(scaled_predict_dataset_x)
# biz kestirim işlemi yaparken çıktıdaki en büyük değerli nöronu tespit etmemiz gerekir.
# bizim en büyük çıktıya sahip olan nöronun çıktı değerinden ziyade 
#   onun çıktıdaki kaçıncı nöron olduğunu kullanarak tahminin hangi Iris olduğunu bulabiliriz.
# Bu işlem NumPy kütüphanesindeki argmax fonksiyonu ile yapılabilir.
predict_indexes = np.argmax(predict_result, axis=1)

# sütun index'lerinin sıralaması ohe.categories_ özniteliğinde mevcut.
for pi in predict_indexes:
    print(ohe.categories_[0][pi]) # ohe.categories_ bize listelerden oluşan bir array verir. o yüzden [0][pi]

"""
predict_categories = ohe.categories_[0][predict_indexes]
print(predict_categories)
"""

#predict-iris.csv
"""
4.8,3.4,1.6,0.2
4.8,3.0,1.4,0.1
4.3,3.0,1.1,0.1
6.6,3.0,4.4,1.4
6.8,2.8,4.8,1.4
6.7,3.0,5.0,1.7
6.0,2.9,4.5,1.5
5.7,2.6,3.5,1.0
6.3,2.5,5.0,1.9
6.5,3.0,5.2,2.0
6.2,3.4,5.4,2.3
5.9,3.0,5.1,1.8
"""