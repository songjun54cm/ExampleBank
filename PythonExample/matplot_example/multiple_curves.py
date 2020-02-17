import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(100))
y1 = 2*x + 1
y2 = 3*x
y3 = x**1.5

# plt.plot(x, y1, 'r')
# plt.plot(x, y2, 'b')
# plt.plot(x, y3, 'g')
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot(x, y3, label='y3')
plt.legend()
plt.show()
