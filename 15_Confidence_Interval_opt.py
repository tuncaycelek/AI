import numpy as np
from scipy.stats import norm

#----------------------------------------------------------------------------------------------------------------------------
#    Aşağıdaki örnekte ortalaması 100, standart sapması 15 olan normal dağılıma uygun rastgeele 1,000,000 değer üretilmiştir.
#    Bu değerlerin anakütleyi oluşturduğu varsayılmıştır. Sonra bu anakütle içerisinden rastgele 60 elemanlık bir örnek elde
#    edilmiştir. Bu örneğe dayanılarak anakütle ortalaması norm nesnesinin interval metoduyla 0.95 güven düzeyiyle elde edilip 
#    ekrana yazdırılmıştır.
#----------------------------------------------------------------------------------------------------------------------------

POPULATION_SIZE = 1_000_000
SAMPLE_SIZE = 60

population = norm.rvs(100, 15, POPULATION_SIZE)

population_mean = np.mean(population)
population_std = np.std(population)

print(f'population mean: {population_mean}')
print(f'population std: {population_std}')

sample = np.random.choice(population, SAMPLE_SIZE)
sample_mean = np.mean(sample)
sampling_mean_std = population_std / np.sqrt(SAMPLE_SIZE)

print(f'sample mean: {sample_mean}')

lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)
print(f'[{lower_bound}, {upper_bound}]')