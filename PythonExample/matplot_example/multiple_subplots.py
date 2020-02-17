import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(100))
y1 = 2*x + 1
y2 = 3*x
y3 = x**2

fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y1, label='y1')
axs[0].plot(x, y2, label='y2')
axs[0].legend()
axs[1].plot(x, y3, label='y3')
axs[1].legend()
plt.show()

