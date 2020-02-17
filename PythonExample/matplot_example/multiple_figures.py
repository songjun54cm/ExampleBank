import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
axs1 = plt.subplot(1, 2, 1)
axs2 = plt.subplot(1, 2, 2)

plt.figure(2)
axs21 = plt.subplot(1, 1, 1)

x = np.array(range(100))
y1 = 2*x + 1
plt.sca(axs1)
plt.plot(x, y1)
plt.legend()

y2 = 3*x
plt.sca(axs2)
plt.plot(x, y2)
plt.legend()

y3 = x**2
plt.sca(axs21)
plt.plot(x, y3)
plt.legend()

plt.show()


