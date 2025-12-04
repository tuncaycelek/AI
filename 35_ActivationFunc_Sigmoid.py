import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid

x = np.linspace(-10, 10, 1000)
#y = np.e ** x / (1 + np.e ** x)
y = sigmoid(x).numpy()

plt.title('Sigmoid (Logistic) Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-1, 1)

plt.plot(x, y)
plt.show()
