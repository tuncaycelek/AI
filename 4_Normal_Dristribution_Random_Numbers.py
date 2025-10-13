import random
import statistics

nd = statistics.NormalDist(100, 15)

#result = [nd.inv_cdf(random.random()) for _ in range(10000)]
#result = nd.samples(10000)
result = [random.gauss(0, 1) for _ in range(10000)]

import matplotlib.pyplot as plt

plt.hist(result, bins=20)
plt.show()
