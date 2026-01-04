import numpy as np
from sklearn.preprocessing import MaxAbsScaler

def manuel_maxabs_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    
    for col in range(dataset.shape[1]):
        maxabs_val = np.max(np.abs(dataset[:, col]))
        scaled_dataset[:, col] = 0 if maxabs_val == 0 else  dataset[:, col] / maxabs_val       
    return scaled_dataset


def maxabs_scaler(dataset):
        return dataset /  np.max(np.abs(dataset), axis=0)


dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])

result = maxabs_scaler(dataset)
result2 = manuel_maxabs_scaler(dataset)
print(dataset)
print(result)
print(result2)

print('------------------------------------------------')

mas = MaxAbsScaler()
mas.fit(dataset)
scaled_dataset = mas.transform(dataset)

print(scaled_dataset)
