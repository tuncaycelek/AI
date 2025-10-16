from scipy.stats import poisson
import matplotlib.pyplot as plt

    #----------------------------------------------------------------------------------------------------------------------------
    #    Aşağıda lamda değeri (ortalaması) 10 olan poisson dağılımı için saçılma grafiği çizdirilmiştir. Bu grafiğin normal dağılım
    #    grafiğini andırmakla birlikte sağdan çarpık olduğuna dikkat ediniz. 
    #----------------------------------------------------------------------------------------------------------------------------

plt.title('Lamda = 10 İçin Poisson Dağılımı', fontweight='bold', pad=10)
x = range(0, 40)
y = poisson.pmf(x, 10)

plt.scatter(x, y)
plt.show()

    #----------------------------------------------------------------------------------------------------------------------------
    #    Poisson dağılımında lamda değeri yüksek tutulduğunda saçılma grafiğinin Gauss eğrisine benzediğine dikkat ediniz. 
    #----------------------------------------------------------------------------------------------------------------------------

from scipy.stats import poisson
import matplotlib.pyplot as plt

plt.title('Lamda = 100 İçin Poisson Dağılımı', fontweight='bold', pad=10)
x = range(0, 200)
y = poisson.pmf(x, 100)

plt.scatter(x, y)
plt.show()

result = poisson.pmf(3, 4)
print(result)