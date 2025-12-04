import sympy
from sympy import init_printing

init_printing()

x = sympy.Symbol('x')
fx = sympy.E ** x / (1 + sympy.E ** x)
dx = sympy.diff(fx, x)

print(dx)

import numpy as np

np.linspace(-10, 10, 1000)
pdx = sympy.lambdify(x, dx)

x = np.linspace(-10, 10, 1000)
y = pdx(x)

import matplotlib.pyplot as plt

plt.title('First Derivative of Sigmoid Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-0.4, 0.4)

plt.plot(x, y)
plt.show()
