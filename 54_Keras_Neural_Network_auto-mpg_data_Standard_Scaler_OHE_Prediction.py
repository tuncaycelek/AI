# "Auto MPG"
# https://archive.ics.uci.edu/dataset/9/auto+mpg

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import pickle

"""
    Burada Auto-MPGveri kümesinde predict işleminin nasıl yapıldığı gösterilmektedir. Burada predict sırasında yeni bir 
    OneHotEncoder nesnesi oluşturulmamış eğitim sırasındaki OneHotEncoder nesnesi  kullanılmıştır. Ayrıca bu örnekte modelde 
    kullanılan nesnelerin disk'te saklandığına da dikkat ediniz.
"""

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
# 7. indexli araba menşei alanını one-hot-encoding'e sokmak için ayırıyoruz.
df_1 = df.iloc[:, :7]
df_2 = df.iloc[:, [7]] # OHE 2 boyutlu bir şey istediği için bu şekilde yapıyoruz. 
                       # df.iloc[:, 7] yapsaydık tek boyutlu Series nesnesi olurdu

# DataFrame olarak devam etmek için numpy'a dönüyoruz.
dataset_1 = df_1.to_numpy()
dataset_2 = df_2.to_numpy()

# veri kümemizde 7.kolonu OHE'a sokuyoruz.
ohe = OneHotEncoder(sparse_output=False)
dataset_2 = ohe.fit_transform(dataset_2)

# ana dataset'im ile OHE uyguladığım dataset'imi birleştiriyorum.
dataset = np.concatenate([dataset_1, dataset_2], axis=1)

dataset_x = dataset[:, 1:]
dataset_y = dataset[:, 0]
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

# Çizimlerimizi yapalım.
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
plt.legend(['Measn Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

# modelimizi save ediyoruz. pickle modülüyle OHE nesnesini bütünsel olarak saklayıp predict aşamasında kullanabiliriz.
# modeli de, ağırlık değerleri de saklanıyor 
model.save('auto-mpg.h5')

with open('auto-mpg.pickle', 'wb') as f:
    pickle.dump((ohe, ss), f)

# prediction

# Daha önce belirlediğim category'leri prediction'da kullanabilirim.
# ama zaten yukarıda bir ohe nesnesi yarattığımız için artık bunu yaratmaya gerek yok.
#ohe_predict = OneHotEncoder(sparse_output=False,categories=[ohe.categories])

predict_df = pd.read_csv('DataSets\\predict.csv', header=None)
predict_df_1 = predict_df.iloc[:, :6]
predict_df_2 = predict_df.iloc[:, [6]]

predict_dataset_1 = predict_df_1.to_numpy()
predict_dataset_2 = predict_df_2.to_numpy()
predict_dataset_2  = ohe.transform(predict_dataset_2)

predict_dataset = np.concatenate([predict_dataset_1, predict_dataset_2], axis=1)
scaled_predict_dataset = ss.transform(predict_dataset)

#artık sonuçlarımızı alalım.
predict_result = model.predict(scaled_predict_dataset)

for val in predict_result[:, 0]:
    print(val)

