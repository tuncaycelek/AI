#Standart sapmanın karesine "varyans (variance)" denilmektedir.
#Varyans değerlerin karelerinin ortalamasının değerlerin ortalamasının 
    #karesinden çıkartılmasıyla da hesaplanabilmektedir:

    
    
import statistics
import numpy as np
import pandas as pd

data = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]

result = statistics.pvariance(data)
print(result)               # 7.666666666666667

result = np.var(data)
print(result)               # 7.666666666666667

s = pd.Series(data)
result = s.var(ddof=0)
print(result)               # 7.666666666666667


#varyans = değerlerin karelerinin ortalaması - değerlerin ortalamasının karesi

total = 0
total_square = 0
count = 0

for x in data:
    total += x
    total_square += x ** 2
    count += 1

v = total_square / count - (total / count) ** 2
print(v)