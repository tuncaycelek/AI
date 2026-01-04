import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import pandas as pd
import re

# imdb veri kümesi yoğun kullanıldığından keras içerisine dahil edilmiştir. 
# biz doğrudan import ederek bu veri kümesini kullanabiliriz.
from tensorflow.keras.datasets import imdb

# load_data fonksiyonu x verileri olarak vektörizasyon sonucundaki vektörleri 
# sözcüklerin indekslerine ilişkin vektörleri bir liste olarak vermektedir
(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = imdb.load_data()

# get_word_index fonksiyonui sözcük haznesini bir sözlük olarak verir
vocab_dict = imdb.get_word_index()
"""
rev_vocab_dict = {value: key for key, value in vocab_dict.items()}
convert_text = lambda text_numbers: ' '.join([rev_vocab_dict[tn - 3] for tn in text_numbers if tn > 2])
"""

# training_dataset_x ve test_dataset_x listelerini aşağıdaki gibi binary vector haline getiriyoruz
# indekslerin bulunduğu liste listesini ve oluşturulacak matrisin sütun uzunluğunu parametre alıyor
def vectorize(sequence, colsize):
    dataset_x = np.zeros((len(sequence), colsize), dtype='uint8')
    for index, vals in enumerate(sequence):
        dataset_x[index, vals] = 1       
    return dataset_x

# load_data fonksiyonu bize listelerdeki sözcük indekslerini üç fazla vermektedir. 
# bu sözcük indekslerindeki 0, 1 ve 2 indeksleri özel anlam ifade etmektedir. 
# bizim bu indekslerden 3 çıkartmamız gerekmektedir. 
training_dataset_x = vectorize(training_dataset_x, len(vocab_dict) + 3)
test_dataset_x = vectorize(test_dataset_x, len(vocab_dict) + 3)

#model
model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

# grafik
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

predict_df = pd.read_csv('DataSets\\predict-imdb.csv')

predict_list = []
for text in predict_df['review']:
    index_list = []
    words = re.findall('[A-Za-z0-9]+', text.lower())
    for word in words:
        index_list.append(vocab_dict[word] + 3)
    predict_list.append(index_list)
    
predict_dataset_x = vectorize(predict_list, len(vocab_dict) + 3)

predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

"""
# dataset_x'teki birinci yorumun yazı haline getirilmesi

rev_vocab_dict = {index: word for word, index in vocab_dict.items()}

word_indices = np.argwhere(training_dataset_x[0] == 1).flatten()
words = [rev_vocab_dict[index - 3] for index in word_indices if index > 2]
text = ' '.join(words)
print(text)
"""    