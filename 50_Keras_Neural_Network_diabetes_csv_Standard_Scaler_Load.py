import numpy as np
from tensorflow.keras.models import load_model
import pickle 

model = load_model('diabetes.h5')

with open('diabetes-scaling.dat', 'rb') as f:
    ss = pickle.load(f)

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

scaled_predict_dataset = ss.transform(predict_dataset)
predict_result = model.predict(scaled_predict_dataset)
print(predict_result)

for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')
    