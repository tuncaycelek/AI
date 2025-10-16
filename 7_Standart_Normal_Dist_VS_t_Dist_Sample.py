import numpy as np
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

    #Aşağıdaki programda standart normal dağılım ile 5 serbestlik derecesi ve 30 serbestlik derecesine ilişkin t dağılımlarının
    #olasılık yoğunluk fonksiyonları çizdirilmiştir. Burada özellikle 30 serbestlik derecesine ilişkin t dağılımının grafiğinin 
    #standart normal dağılım grafiği ile örtüşmeye başladığına dikkat ediniz. 
    
plt.figure(figsize=(15, 10))
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_xticks(range(-4, 5))
plt.plot(x, y)

y = t.pdf(x, 5)
plt.plot(x, y)

y = t.pdf(x, 30)
plt.plot(x, y, color='red')

plt.legend(['Standart Normal Dağılım', 't Dağılımı (Serbestlik Derecesi = 5)', 
            't dağılımı (Serbestlik Derecesi = 30)'])

plt.show()

    #Tabii standart normal dağılımla t dağılımının olasılık yoğunluk fonksiyonları farklı olduğuna göre aynı değerlere ilişkin 
    #kümülatif olasılık değerleri de farklı olacaktır. 
    
x = [-0.5, 0, 1, 1.25]
result = norm.cdf(x)
print(result)               # [0.30853754 0.5        0.84134475 0.89435023]

x = [-0.5, 0, 1, 1.25]
result = t.cdf(x, 5)    
print(result)               # [0.31914944 0.5        0.81839127 0.86669189]