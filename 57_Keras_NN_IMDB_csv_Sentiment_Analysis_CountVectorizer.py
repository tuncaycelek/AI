import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('DataSets\\IMDB Dataset.csv')

cv = CountVectorizer(dtype='uint8', binary=False)

cv.fit(df['review'])
dataset_x = cv.transform(df['review']).todense()


# df içerisindeki son kolon olan sentiment kolonundan yorumun olumlu/olumsuz bilgisini y olarak alıyoruz. 
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')

# dataset imizi x/y, training/test olarak split ediyoruz.    
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

# modelimizi oluşturabiliriz.
model = Sequential(name='IMDB')

# girdi katmanında çok fazla nöron olduğu için katmanlardaki nöron sayılarını yükseltebiliriz.
model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1')) 
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

# parametre sayısı çok yüksek olduğu için epoch'u düşük tutuyoruz.
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

# histogramlarımızı çizelim.
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

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
 
# prediction
predict_df = pd.read_csv('DataSets\\predict-imdb.csv')

predict_dataset_x = cv.transform(predict_df['review']).todense()
    
predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

"""
# dataset_x'teki birinci yorumun yazı haline getirilmesi
rev_vocab_dict = {index: word for word, index in cv.vocabulary_.items()}

word_indices = np.argwhere(dataset_x[0] == 1)[:, 1] 
words = [rev_vocab_dict[index] for index in word_indices]
text = ' '.join(words)
print(text)
"""


    