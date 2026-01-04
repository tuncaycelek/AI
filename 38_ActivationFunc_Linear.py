import numpy as np
import matplotlib.pyplot as plt

def linear(x):
      return x

x = np.linspace(-10, 10, 1000)
y = linear(x)

"""
from tensorflow.keras.activations import linear

y = linear(x).numpy()
"""

plt.title('Linear Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
plt.plot(x, y, color='red')
plt.show()