import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3)

N = 100     # data set size
d = 2       # 2 classes of data

# Function to w and Y
def dataGen(X):
    w = np.random.uniform(-1, 1, size=(X.shape[1]))
    Y = np.sign(np.dot(X, w))
    return w, Y

# Generate random training data
X = np.random.uniform(-1, 1, size=(N, d+1))
X[:, 0] = 1

# Create 5-d data Z by mapping
# (1, x1, x2, x1^2, x2^2, x1*x2)
Z = np.random.uniform(-1, 1, size=(N, 6))
Z[:, 0] = 1
for i in range(len(X)):
    Z[i][0] = 1
    Z[i][1] = X[i][1]
    Z[i][2] = X[i][2]
    Z[i][3] = np.power(X[i][1], 2)
    Z[i][4] = np.power(X[i][2], 2)
    Z[i][5] = np.multiply(X[i][1], X[i][2])


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

# Generate / Plot Data
twoD = dataGen(X)
w = twoD[0]
Y = twoD[1]
ind_pos = np.where(Y == 1)[0]
ind_neg = np.where(Y == -1)[0]
axs[0].plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')    # red dot points
axs[0].plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')    # blue 'x' points

# 2D Perceptron
l = np.linspace(-1, 1, 50)
f = (-(w[1]/w[2]) * l) - w[0]/w[2]
axs[0].plot(l, f, label='PLA 2D target fxn', color='red')
pla = perceptron(X, Y, pocket=False)
g = pla[0]
ein_pla = pla[1]
g_x = (-(g[1]*X)/g[2]) - (g[0]/g[2])
axs[0].plot(X, g_x, label='PLA final hypothesis', color='blue')
axs[1].plot(np.linspace(-1, 1, 1000), ein_pla, label='Ein PLA 2D', color='green')

# 2D Linear Regression
Xdag = np.matmul(np.linalg.pinv(np.matmul(X.transpose(), X)), X.transpose())
w1 = np.matmul(Xdag, Y)
wTx = np.matmul(w1, X.transpose())
xw = np.matmul(X, w1)
p1 = np.matmul(wTx, xw)
p2 = np.multiply(np.matmul(wTx, Y), 2)
p3 = np.matmul(Y.transpose(), Y)
ein_lin_reg = np.add(np.subtract(p1, p2), p3)/100
print("Ein Linear Regression: " + str(ein_lin_reg))
lin_reg = (-(w1[0])/w1[2]) + (-w1[1]*X/w1[2])
axs[0].plot(X, lin_reg, label='Linear Regression', color='purple')

# 5D Perceptron
fiveD = dataGen(Z)
w = fiveD[0]
Y = fiveD[1]
pla_5 = perceptron(X, Y, pocket=False)
g1 = pla_5[0]
ein_pla5 = pla_5[1]
pla5_g = (-g1[0]/g1[2]) - (g1[1]*X/g1[2])
print("Ein 5D Perceptron: " + str(np.sum(ein_pla5)/1000))
itr = np.linspace(0, 1000, 1000)
axs[2].plot(itr, ein_pla5, label='Ein PLA 5D', color='red')

axs[1].legend(loc='best')
axs[2].legend(loc='best')

plt.show()
