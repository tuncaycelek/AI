import numpy as np
from scipy.stats import t, sem

#----------------------------------------------------------------------------------------------------------------------------
#    Anakütlenin standart sapmasının da bilinmediğini varsayalım. Bu değerlerden hareketle %95 güven düzeyinde güven aralığını 
#    kullanarak oluşturuyoru. örneğin standart sapmasını hesaplarken ddof=1 kullandığımıza dikkat ediniz. Güven aralıkları normal dağılım 
#    kullanılarak değil t dağılımı kullanılarak elde edilmiştir. t dağılımındaki serbestlik derecesinin (ppf fonksiyonun ikinci parametresi) 
#    örnek büyüklüğünün bir eksik değeri olarak alındığına dikkat ediniz.
#----------------------------------------------------------------------------------------------------------------------------
        
sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  
                   87.99167058,  95.16206704, 101.78211828,  80.71674993, 126.3793041 , 105.07860807, 
                   98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 117.31491413, 
                   105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 
                   81.77519378, 117.61282039, 109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 
                   100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sampling_mean_std = sample_std / np.sqrt(len(sample))

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')


#----------------------------------------------------------------------------------------------------------------------------
#    Örneklem dağılımının standart sapmasına standart hata denildiğini anımsayınız. Aslında scipy.stats modülünde bu değeri hesaplayan 
#    sem isimli bir fonksiyon da bulunmaktadır. 
#----------------------------------------------------------------------------------------------------------------------------  

sampling_mean_std = sem(sample)

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')

#----------------------------------------------------------------------------------------------------------------------------
#   t nesnesinin de ilişkin olduğu sınıfın interval isimli bir metodu bulunmaktadır. Bu metot zaten doğrudan t dağılımını kullanarak 
#   güven aralıklarını hesaplamaktadır. interval(confidence, df, loc=0, scale=1) confidence parametresi yine "güven düzeyini (confidence level)",
#   df parametresi serbestlik derecesini, loc ve scale parametreleri de sırasıyla ortalama ve standart sapma değerlerini belirtmektedir
#   Burada loc parametresine biz örneğimizin ortalamasını, scale parametresine de örneklem dağılımının standart sapmasını girmeliyiz. 
#   Tabii örneklem dağılımının standart sapması yine örnekten hareketle elde edilecektir. 
#----------------------------------------------------------------------------------------------------------------------------

lower_bound, upper_bound = t.interval(0.95, len(sample) - 1, sample_mean, sampling_mean_std)
print(f'[{lower_bound}, {upper_bound}]')