import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import relu

def relu(x):
      return np.maximum(x, 0)  

x = np.linspace(-10, 10, 1000)
#y = relu(x)
y = relu(x).numpy()

plt.title('Relu Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
plt.plot(x, y, color='red')
plt.show()
