import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.e ** (2 * x) - 1) / (np.e ** (2 * x) + 1)

x = np.linspace(-10, 10, 1000)
y = tanh(x)

#from tensorflow.keras.activations import tanh
#y = tanh(x).numpy()

plt.title('Hiperbolik Tanjant (tanh) Fonksiyonunun Grafiği', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-1, 1)

plt.plot(x, y)
plt.show()