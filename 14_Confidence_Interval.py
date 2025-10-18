import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109

sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')              # [105.20454606435501, 112.79545393564499]

#Burada biz anakütlenin standart sapmasını bildiğimiz için örnek ortalamalarına ilişkin normal dağılımın standart sapmasını
#hesaplayabildik. Buradan elde ettiğimiz güven aralığı şöyle olmaktadır:

# [105.20454606435501, 112.79545393564499]

#Güven düzeyini yükseltirsek güven aralığının genişleyeceği açıktır. Örneğin bu problem için güven düzeyini %99 olarak 
#belirlemiş olalım:


sample_size = 60
population_std = 15
sample_mean = 109
sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.005, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.995,  sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')             # [104.01192800234102, 113.98807199765896]


print(f'{lower_bound}, {upper_bound}')      # 104.01192800234102, 113.98807199765896

#----------------------------------------------------------------------------------------------------------------------------
#   örneği büyüttüğümüzde güven aralıkları daralmakta ve anakütle ortalaması daha iyi tahmin 
#    edilmektedir. Örnek büyüklüğünün artırılması belli bir noktaya kadar aralığı iyi bir biçimde daraltıyorsa da belli bir
#    noktadan sonra bu daralma azalmaya başlamaktadır. Örneklerin elde edilmesinin belli bir çaba gerektirdiği durumda örnek
#    büyüklüğünün makul seçilmesi önemli olmaktadır.
#----------------------------------------------------------------------------------------------------------------------------    

population_std = 15
sample_mean = 109

for sample_size in range(30, 105, 5):
    sampling_mean_std = population_std / np.sqrt(sample_size)

    lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
    upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)

    print(f'sample size: {sample_size}: [{lower_bound}, {upper_bound}]')     

#----------------------------------------------------------------------------------------------------------------------------
#    anakütle ortalamaları aslında tek hamlede norm nesnesinin ilişkin olduğu sınıfın interval metoduyla da
#    elde edilebilmektedir.
#----------------------------------------------------------------------------------------------------------------------------

sample_size = 60
population_std = 15
sample_mean = 109

sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')     

lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')    


