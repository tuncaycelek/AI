import numpy as np
import pandas as pd
import re #regular expressions module
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('DataSets\\IMDB Dataset.csv')

# boş bir küme oluşturuyoruz. bu küme dataset içerisindeki kelimeleri unique bir şekilde elde etmemizi sağlayacak.
vocab = set()

for text in df['review']:
    words = re.findall('[a-zA-Z0-9]+', text.lower()) #sadece ingilizce regex
    vocab.update(words)

# uniq hale getirdiğimiz sözlüğümüzü index'liyoruz.
vocab_dict = {}
for index, word in enumerate(vocab):
    vocab_dict[word] = index
#vocab_dict = {word: index for index, word in enumerate(vocab)}
   
    
# ihtiyacımız olan uzunlukta içi sıfırlarla dolu matris
dataset_x = np.zeros((len(df), len(vocab)), dtype='uint8')
# df içerisindeki son kolon olan sentiment kolonundan yorumun olumlu/olumsuz bilgisini y olarak alıyoruz. 
dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8')

for row, text in enumerate(df['review']):
    words = re.findall('[a-zA-Z0-9]+', text.lower()) # her bir satırdaki kelimeler
    #print([word for word in words]); print('\n')
    word_numbers = [vocab_dict[word] for word in words] # o kelimelerin vocab_dict'deki index karşılığı 
    dataset_x[row, word_numbers] = 1 # dataset_x'de ilgili satırdaki kelimelerin index karşılıklarının 1'lenmesi

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


predict_dataset_x = np.zeros((len(predict_df), len(vocab)))
for row, text in enumerate(predict_df['review']):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    predict_dataset_x[row, word_numbers] = 1
    
predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

"""
# dataset_x'teki birinci yorumun yazı haline getirilmesi
rev_vocab_dict = {index: word for word, index in vocab_dict.items()}

word_indices = np.argwhere(dataset_x[0] == 1).flatten()
words = [rev_vocab_dict[index] for index in word_indices]
text = ' '.join(words)
print(text)
""" 



    