from scipy.stats import norm, uniform, kstest


#    Aşağıdaki örnekte normal dağılmış bir anakütleden ve düzgün dağılmış bir anakütleden rastgele örnekler çekilip kstest
#    fonksiyonuna sokulmuştur. Burada birinci örnek için p değeri 1’e yakın, ikinci örnek için p değeri 0’a çok yakın çıkmıştır. 
#    Böylece birinci örneğin alındığı anakütlenin normal dağılmış olduğu ikinci örneğin alındığı anakütlenin ise normal 
#    dağılmamış olduğu söylenebilir. 
#----------------------------------------------------------------------------------------------------------------------------


sample_norm = norm.rvs(size=1000)

result = kstest(sample_norm, 'norm')
print(result.pvalue)        # 1'e yakın bir değer             

sample_uniform = uniform.rvs(size=1000) 

result = kstest(sample_uniform, norm.cdf)
print(result.pvalue)        # 0'a çok yakın bir değer

#    Örneğin biz ortalaması 100 standart 
#    sapması 15 olan bir dağılıma ilişkin test yapmak isteyelim. Bu durumda test aşağıdaki gibi yapılmalıdır. 
#----------------------------------------------------------------------------------------------------------------------------
sample_norm = norm.rvs(100, 15, size=1000)
result_norm = kstest(sample_norm, 'norm', args=(100, 15)) 

sample_uniform = uniform.rvs(100, 100, size=1000) 
result_uniform = kstest(sample_uniform, 'norm', args=(100, 15))

import matplotlib.pyplot as plt 

plt.figure(figsize=(20, 8))
ax1 = plt.subplot(1, 2, 1)
ax1.set_title(f'p değeri: {result_norm.pvalue}', pad=15, fontweight='bold')

ax2 = plt.subplot(1, 2, 2)
ax2.set_title(f'p değeri: {result_uniform.pvalue}', pad=15, fontweight='bold')
ax1.hist(sample_norm, bins=20)

ax2.hist(sample_uniform, bins=20)
plt.show()