import pickle
import numpy as np
import pandas as pd
"""
from tensorflow.keras.models import load_model

# prediction

model = load_model('auto-mpg.h5')
"""
	#Düzeltme : bu kodu çalıştırdığım zaman load_model fonksiyonu mse ile ilgili hata verdiğinden dolayı bir fix yaptım : 
	
	# fix begin : 
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model(
    "auto-mpg.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)	
	# fix end : 
	
with open('auto-mpg.pickle', 'rb') as f:
    ohe, ss = pickle.load(f)    

# prediction
predict_df = pd.read_csv('DataSets\\predict.csv', header=None)

predict_df_1 = predict_df.iloc[:, :6]
predict_df_2 = predict_df.iloc[:, [6]]

predict_dataset_1 = predict_df_1.to_numpy()
predict_dataset_2 = predict_df_2.to_numpy()
predict_dataset_2  = ohe.transform(predict_dataset_2)

predict_dataset = np.concatenate([predict_dataset_1, predict_dataset_2], axis=1)
scaled_predict_dataset = ss.transform(predict_dataset)
predict_result = model.predict(scaled_predict_dataset)

for val in predict_result[:, 0]:
    print(val)