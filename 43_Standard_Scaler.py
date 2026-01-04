import numpy as np
from sklearn.preprocessing import StandardScaler


data= np.array([4, 7, 1, 90, 45, 70, 23, 12, 6, 9, 45, 82, 65])
mean = np.mean(data)
std = np.std(data)
print(f'Mean: {mean}, Standard Deviation: {std}')

result = (data - mean) / std

print(f'Data: {data}')
print(f'Scaled Data: {result}')

print('-' * 30)

dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])

# NumPy'ın eksensel işlem yapma özelliği ile tek satırda yapabiliriz
def standard_scalerN(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    
def standard_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    for col in range(dataset.shape[1]):
        scaled_dataset[:, col] = (dataset[:, col] - np.mean(dataset[:, col])) / np.std(dataset[:, col])
    return scaled_dataset

result = standard_scaler(dataset)     
print(dataset)
print(result)
result = standard_scalerN(dataset)     
print(result)
# sklearn.preprocessing içerisindeki StandardScaler ile tek hamlede yapabiliriz.
ss = StandardScaler()
scaled_dataset = ss.fit_transform(dataset)
print(scaled_dataset)
