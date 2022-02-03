import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2)

X = np.array([[0, 0], [2, 2], [2, 0]])
X1 = np.array(np.transpose([0, 2, 2]))
X2 = np.array(np.transpose([0, 2, 0]))
y = np.array(np.transpose([-1, -1, 1]))
w1 = np.array(np.transpose([1.2, -3.2]))
w2 = np.array(np.transpose([2.4, -6.4]))
b1 = -0.5
b2 = -1

h1 = np.sign(w1[0]*X1 + w1[1]*X2 + b1)
axs[0].plot(h1, 'r', label='H1')

h2 = np.sign(w2[0]*X1 + w2[1]*X2 + b2)
axs[1].plot(h2, 'g', label='H2')

axs[0].legend(loc='best')
axs[1].legend(loc='best')
plt.show()
