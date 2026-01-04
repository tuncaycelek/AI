# "Auto MPG"
# https://archive.ics.uci.edu/dataset/9/auto+mpg

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('DataSets\\auto-mpg.data', delimiter=r'\s+', header=None)

#araba markalarının bulunduğu son sütunu atıyoruz.
df = df.iloc[:, :-1] # df.drop(8, axis=1, inplace=True) #(veya bunu da kullanabilirdik.)
# 3. indexli sütundaki ? işareti olanlara numpy null -> np.nan atıyoruz.
df.iloc[df.loc[:, 3] == '?', 3] = np.nan
# 3. indexli sütunun tipini belirliyoruz. Çünkü içinde ? olduğu zaman pandas bunu nümerik bir alan olarak değerlendiremiyor.
df[3] = df[3].astype('float64')
# 3. indexli sütunda np.nan değerleri için Simple Imputer uyguluyoruz.
si = SimpleImputer(strategy='mean', missing_values=np.nan)
df[3] = si.fit_transform(df[[3]])
# 7. indexli araba menşei alanını one-hot-encoding'e sokuyoruz.
df = pd.get_dummies(df, columns=[7], dtype='uint8')
# artık dataset'imizi numpy'a çekebiliriz.
dataset = df.to_numpy()
# veri kümesini x ve y olarak ayrıştırıyoruz.
dataset_x = dataset[:, 1:]
dataset_y = dataset[:, 0]

#df.hist()
#veri kümemizi eğitim ve test amacıyla train_test_split fonkisyonu ile ayrıştırıyoruz.
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)
        

# scikit-learn kullanarak özellik ölçeklemesi yapıyoruz. 
ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)
scaled_test_dataset_x = ss.transform(test_dataset_x)

# Buradan sonra iki saklı katman kullarak standart olarak modelimizi kuruyoruz ve eğitimimizi yapıyoruz.       
# Saklı katmanlardaki aktivasyon fonksiyonlarını yine "relu" olarak alacağız. 
# Ancak çıktı katmanındaki aktivasyonun "linear" olduğuna dikkat edelim.
model = Sequential(name='Auto-MPG')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(32, activation='relu', name='Hidden-1'))
model.add(Dense(32, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

# Şimdi de modelimizi compile edip fit işlemi uygulayalım. Modelimiz için optimizasyon algoritması yine "rmsprop" seçilebilir. 
# Regresyon problemleri için loss fonksiyonu genellikle "mean_squared_error" biçiminde alınabilir.
# Yine regresyon problemleri için "mean_absolute_error" metrik değeri kullanılabilir.
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

#Modelimizi test veri kümesiyle test edebiliriz
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Mean Absolute Error', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

# Kestirim yapılacak veriyi predict.csv dosyasının içinden alalım.
predict_df = pd.read_csv('DataSets\\predict.csv', header=None)
""" # Dosya içeriği :
8,307.0,130.0,3504,12.0,70,1
4,350.0,165.0,3693,11.5,77,2
8,318.0,150.0,3436,11.0,74,3
"""
# Kestirilecek veriler üzerinde de one-hot-encoding dönüştürmesinin ve özellik ölçeklemesinin yapılması gerekir
predict_df = pd.get_dummies(predict_df, columns=[6])
predict_dataset_x = predict_df.to_numpy() 
scaled_predict_dataset_x = ss.transform(predict_dataset_x)

# prediction : Şimdi kestirim yapalım.
predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    
