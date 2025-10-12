#Standart sapma, değerlerin ortalamadan farklarının karelerinin 
    #ortalamasının karekökü alınarak hesaplanmaktadır.
    
import numpy as np

data1 = np.array([2, 5, 7, 5, 5, 4, 5, 5, 6, 6])
mean1 = np.mean(data1)
std1 = np.std(data1)
print(f'mean : {mean1}, std: {std1}')

data2 = np.array([1, 5, 7, 1, 9, 4, 5, 5, 6, 7])
mean2 = np.mean(data2)
std2 = np.std(data2)
print(f'mean : {mean2}, std: {std2}')

data3 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
mean3 = np.mean(data3)
std3 = np.std(data3)
print(f'mean : {mean3}, std: {std3}')

# mean : 5.0, std: 1.2649110640673518
#mean : 5.0, std: 2.4083189157584592
#mean : 5.0, std: 0.0