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
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import Callback

        
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

# -------------------------------------------------- Artık Modelimizi kurup, eğitiyoruz.
# 1. Öncelikle bir model nesnesi oluşturulmalıdır.
model = Sequential(name='Diabetes')

# 2. Model nesnesinin yaratılmasından sonra katman nesnelerinin oluşturulup model nesnesine eklenmesi gerekir.
model.add(Input((training_dataset_x.shape[1],)))                                    # model'imize input'umuzu veriyoruz.

hidden1 = Dense(16, activation='relu', name='Hidden-1')                             # 1. hidden katmanı oluşturduk.                          
model.add(hidden1)                                                                  # model nesnesine ekledik.

model.add(Dense(16, activation='relu', name='Hidden-2'))                            # 2. Hidden katman
model.add(Dense(1, activation='sigmoid', name='Output'))                            # çıktı sigmoid layer
                                                                                    # çıktı katmanındaki nöron sayısı, elde etmek istediğimiz
                                                                                    # sonuca göre olmalı.
                                                                                    # ikili sınıflandırma problemlerinde çıktı katmanındaki nöron
                                                                                    # sayısı 1 olur ve aktivasyon fonksiyonu sigmoid olur.
                                                                                    # Sigmoid fonksiyonu 0 ile 1 arasında bir değer vermektedir
# 3. Modele katmanlar eklendikten sonra bir özet bilgi yazdırılabilir.
model.summary()                                                                     # model özeti

# 4. Model oluşturulduktan sonra modelin derlenmesi (compile edilmesi) gerekir.
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy']) # eğitim parametreleri


    # Seçenek 1 : 
# ---------------------- Buraya dikkat, Lambda callback sınıfını kullanıyoruz. ---------------------- 

    # her epoch sonrasında "loss" ve "val_loss" değerleri callback fonksiyonda yazdırılmıştır. Ayrıca her 
    # epoch içerisindeki batch işlemlerinin sonucunda elde edilen "loss" değerleri de ekrana yazdırılmıştır. Aynı zamanda burada 
    # batch'lerdeki loss değerleri bir listede saklanmış ve bu loss değerlerinin ortalaması da ekrana yazdırılmıştır.
"""
batch_losses= []

def on_epoch_begin_proc(epoch, logs):
    global batch_losses
    batch_losses = []
    print(f'eopch: {epoch}')    

def on_epoch_end_proc(epoch, logs):
    loss = logs['loss']
    val_loss = logs['val_loss']
    print(f'\nepoch: {epoch}, loss: {loss}, val_loss: {val_loss}')
    print(f'batch mean: {np.mean(batch_losses)}')
    print('-' * 30)
  
def on_batch_end_proc(batch, logs):
    global total
    loss = logs['loss']
    batch_losses.append(loss)
    print(f'\t\tbatch: {batch}, loss: {loss}')
    
lambda_callback = LambdaCallback(on_epoch_begin=on_epoch_begin_proc, on_epoch_end=on_epoch_end_proc,  on_batch_end=on_batch_end_proc)

# 5. Model derlenip çeşitli belirlemeler yapıldıktan sonra artık gerçekten eğitim aşamasına geçilir.
history_callback = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2, callbacks=[lambda_callback], verbose='auto')
"""
# ---------------------- Buraya dikkat, Lambda callback sınıfını kullanıyoruz. ---------------------- 


    # Seçenek 2 : 
# ---------------------- Buraya dikkat, MyLambdaCallback callback sınıfını kullanıyoruz. ---------------------- 

#   Aslında LambdaCallback sınıfını aslında basit bir biçimde biz de yazabiliriz. Burada yapacağımız şey sınıfın __init__ metodunda 
#   bize verilen fonksiyonları nesnenin özniteliklerinde saklamak ve override ettiğimiz metotlarda bunları çağırmaktır.    
#   MyLambdaCallback sınıfı orijinal LambdaCallback sınıfının aynısı değildir. 
#   Burada yalnızca bu sınıfın nasıl yazıldığına ilişkin bir fikir vermeye çalışıyoruz.

class MyLambdaCallback(Callback):
    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None,  on_batch_end=None):
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        
    def on_epoch_begin(self, epoch, logs):
        if self._on_epoch_begin:
            self._on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch, logs):
        if self._on_batch_begin:
            self._on_batch_begin(batch, logs)
                
    def on_batch_end(self, batch, logs):
        if self._on_batch_end:
            self._on_batch_end(batch, logs)

batch_losses= []

def on_epoch_begin_proc(epoch, logs):
    global batch_losses
    batch_losses = []
    print(f'eopch: {epoch}')    

def on_epoch_end_proc(epoch, logs):
    loss = logs['loss']
    val_loss = logs['val_loss']
    print(f'\nepoch: {epoch}, loss: {loss}, val_loss: {val_loss}')
    print(f'batch mean: {np.mean(batch_losses)}')
    print('-' * 30)
  
def on_batch_end_proc(batch, logs):
    global total
    loss = logs['loss']
    batch_losses.append(loss)
    print(f'\t\tbatch: {batch}, loss: {loss}')

mylambda_callback = MyLambdaCallback(on_epoch_begin=on_epoch_begin_proc, on_epoch_end=on_epoch_end_proc, on_batch_end=on_batch_end_proc)

history_callback = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2, callbacks=[mylambda_callback], verbose=0)            

# ---------------------- Buraya dikkat, MyLambdaCallback callback sınıfını kullanıyoruz. ---------------------- 




# 6. fit işleminden sonra artık model eğitilmiştir. Onun test veri kümesiyle test edilmesi gerekir.
eval_result = model.evaluate(test_dataset_x, test_dataset_y)

# eğitim tamamlandı. eğitim sonuçlarını yazdırıyorum.
for i in range(len(eval_result)):
        print(f'{model.metrics_names[i]}: {eval_result[i]}')
                                        
#print(type(history_callback))
# "diabetes" örneği için fit metodunun geri döndürdüğü History callback nesnesi kullanılarak 
# epoch grafikleri
plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(history_callback.epoch, history_callback.history['loss'])
plt.plot(history_callback.epoch, history_callback.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(history_callback.epoch, history_callback.history['binary_accuracy'])
plt.plot(history_callback.epoch, history_callback.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')



    






