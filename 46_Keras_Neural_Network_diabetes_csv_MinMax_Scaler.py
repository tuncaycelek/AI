# "Pima Indians Diabetes Database"
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# https://github.com/tuncaycelek/AI/blob/main/DataSets/diabetes.csv

# 1. Öncelikle bir model nesnesi oluşturulmalıdır.
# 2. Model nesnesinin yaratılmasından sonra katman nesnelerinin oluşturulup model nesnesine eklenmesi gerekir.
# 3. Modele katmanlar eklendikten sonra bir özet bilgi yazdırılabilir.
# 4. Model oluşturulduktan sonra modelin derlenmesi (compile edilmesi) gerekir.
# 5. Model derlenip çeşitli belirlemeler yapıldıktan sonra artık gerçekten eğitim aşamasına geçilir.
# 6. fit işleminden sonra artık model eğitilmiştir. Onun test veri kümesiyle test edilmesi gerekir.
# 7. Artık model test de edilmiştir. Şimdi sıra "kestirim (prediction)" yapmaya gelmiştir.
# 8. Ve son aşama, sonuçların gösterilmesi !
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
        
TEST_RATIO = 0.2
# her eğitimde ve test işleminde aynı sonucu elde etmek için random tohum değerini sabitlemeye çalışıyoruz.
# reproducible
#np.random.seed(1234567)
#set_random_seed(678901)
df = pd.read_csv('DataSets\\diabetes.csv')

# Eksik veri 
# print((df == 0).sum())

# Imputing
si = SimpleImputer(strategy='mean', missing_values=0) # missing_values=0 parametresine dikkat
impute_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[impute_features] = si.fit_transform(df[impute_features])

# to Numpy
dataset = df.to_numpy()

# dataset_x ve dataset_y oluşturuldu.
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

# training ve test olarak dataset ayrıştırıldı.
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=TEST_RATIO)

# MinMax Scaler kullanarak datayı scale ediyoruz.
mms = MinMaxScaler()

mms.fit(training_dataset_x)
scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)


model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)
eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y, batch_size=32)

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

scaled_predict_dataset = mms.transform(predict_dataset)
predict_result = model.predict(scaled_predict_dataset)
print(predict_result)

for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')

    