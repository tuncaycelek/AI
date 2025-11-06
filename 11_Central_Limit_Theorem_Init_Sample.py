#    Aşağıdaki örnekte 0 ile 1,000,000 arasındaki sayılardan oluşan 1,000,000 elemanlık düzgün dağılmış anakütle içerisinden 
#    100,000 tane 50'lik rastgele örnekler elde edilmiştir. Sonra da elde edilen bu örneklerin ortalamaları hesaplanmış ve bu
#    ortalamaların histogramı çizdirilmiştir. Örnek ortalamalarının dağılımına ilişkin histogramın normal dağılıma benzediğine 
#    dikkat ediniz. 
#----------------------------------------------------------------------------------------------------------------------------

import numpy as np

POPULATION_RANGE = 1_000_000
NSAMPLES = 100_000
SAMPLE_SIZE = 50

samples = np.random.randint(0, POPULATION_RANGE + 1, (NSAMPLES, SAMPLE_SIZE))
samples_means = np.mean(samples, axis=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title('Merkezi Limit Teoremi', fontweight='bold', pad=10)
plt.hist(samples_means, bins=50)

plt.show()

#----------------------------------------------------------------------------------------------------------------------------
#    Tabii yukarıdaki örneği hiç NumPy kullanmadan tamamen Python standart kütüphanesi ile de yapabilirdik.
#----------------------------------------------------------------------------------------------------------------------------

import random
import statistics

POPULATION_RANGE = 1_000_000
NSAMPLES = 100_000
SAMPLE_SIZE = 50

samples_means = [statistics.mean(random.sample(range(POPULATION_RANGE + 1), SAMPLE_SIZE)) for _ in range(NSAMPLES)]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title('Merkezi Limit Teoremi', fontweight='bold', pad=10)
plt.hist(samples_means, bins=50)

plt.show()

#----------------------------------------------------------------------------------------------------------------------------
#    Aşağıdaki örnekte örnek ortalamalarına ilişkin dağılımın standart sapmasının merkezi limit teoreminde belirtilen durum ile 
#    uygunluğu gösterilmektedir. Bu bu örnekte de aslında biz anakütle içerisinden az sayıda örnek aldığımız için olması gereken
#    değerlerden bir sapma söz konusu olacaktır.  Örnek ortalamanın standart sapması = Ana kütle standart sapması / Örnek büyüklüğü karekökü (Standart hata)
#----------------------------------------------------------------------------------------------------------------------------

POPULATION_RANGE = 1_000_000
NSAMPLES = 100_000
SAMPLE_SIZE = 50

population_mean = np.mean(range(POPULATION_RANGE + 1))
population_std = np.std(range(POPULATION_RANGE + 1))

samples = np.random.randint(1, POPULATION_RANGE + 1, (NSAMPLES, SAMPLE_SIZE))
samples_means = np.mean(samples, axis=1)
samples_means_mean = np.mean(samples_means)
sample_means_std = np.std(samples_means)

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının Ortalaması: {samples_means_mean}')
print(f'Fark: {np.abs(population_mean - samples_means_mean)}')

print(f'Merkezi limit teroreminden elde edilen örnek ortalamalarının standart sapması: {population_std / np.sqrt(50)}')
print(f'Örnek ortalamalarının standart sapması: {sample_means_std}')


