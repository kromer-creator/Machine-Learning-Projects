import numpy as np
import matplotlib.pyplot as plt

# 1. Linear regression for classifying noisy data

fig, axs = plt.subplots(3)
N = 100     # data set size
d = 2       # 2 classes of data

# Generate random training data
X = np.random.uniform(-1, 1, size=(N, d+1))
X[:, 0] = 1
# Calculate weights vector
w = np.random.uniform(-1, 1, size=(d+1))
# Compute true labels for the training data
Y = np.sign(np.dot(X, w))

# Add Noise
n = np.random.randint(0, 100, 1)
Y[n[0]] = Y[n[0]]*-1

ind_pos = np.where(Y == 1)[0]   # positive examples
ind_neg = np.where(Y == -1)[0]  # negative examples

# Plot points
axs[0].plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')    # red dot points
axs[0].plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')    # blue 'x' points

# Generate random target function: f(x) = w^Tx
X2 = (-w[1]/w[2]*X) - w[0]/w[2]
line_x = np.linspace(-1, 1, 100)
# Plot target function
axs[0].plot(X, X2, label='Target fxn', color='yellow')

# Perceptron algorithm
def perceptron(X, Y, pocket=False):
    # List of in-sample errors
    eins = []
    best_ein = 1000
    # To be trained
    w_train = np.random.uniform(-1, 1, size=(d + 1))
    best_w = w_train

    for i in range(1000):
        Y1 = np.sign(np.dot(X, w_train))
        # Check for misclassified point
        j = np.random.randint(0, 100, 1)

        if Y1[j] != Y[j]:
            # Update hypothesis
            w_train += X[j][0]*Y[j]

        # Calculate error
        ein = np.count_nonzero(Y1-Y) / 100
        if pocket:
            if ein < best_ein:
                best_w = w_train
                best_ein = ein  ###
        eins.append(best_ein if pocket else ein)

    return best_w if pocket else w_train, eins


# Run PLA
l = np.linspace(0, 1000, 1000)
result = perceptron(X, Y, pocket=False)
g = result[0]
g_x = -(g[0]/g[2]) + (-(g[1]*X)/g[2])
ein_pla = result[1]
axs[0].plot(X, g_x, label='PLA final hypothesis', color='orange')
axs[1].plot(l, ein_pla, label='Ein PLA', color='green')

# Run Pocket Algorithm
result2 = perceptron(X, Y, pocket=True)
g2 = result2[0]
g_x2 = -(g2[0]/g2[2]) + (-(g2[1]*X)/g2[2])
ein_pocket = result2[1]
axs[0].plot(X, g_x2, label='Pocket final hypothesis', color='blue')
axs[2].plot(l, ein_pocket, label='Ein Pocket', color='red')

# Linear Regression     eqn 3.4
Xdag = np.matmul(np.linalg.pinv(np.matmul(X.transpose(), X)), X.transpose())
w = np.matmul(Xdag, Y)
wTx = np.matmul(w, X.transpose())
xw = np.matmul(X, w)
p1 = np.matmul(wTx, xw)
p2 = np.multiply(np.matmul(wTx, Y), 2)
p3 = np.matmul(Y.transpose(), Y)
ein_lin_reg = np.add(np.subtract(p1, p2), p3)/100
print(ein_lin_reg)

# Plot Linear Regression
lin_reg = (-(w[0])/w[2]) + (-w[1]*X/w[2])
axs[0].plot(X, lin_reg, label='Linear Regression', color='purple')

plt.show()



