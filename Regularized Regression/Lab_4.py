import numpy as np
import matplotlib.pyplot as plt

# 4-D Map Function
def mapL(X):
    Z = np.dstack((np.ones(X.shape), X, 0.5 * (3 * (X ** 2) - 1), 0.5 * ((5 * X ** 3) - 3 * X),
                   0.125 * ((35 * X ** 4) - (30 * X ** 2) + 3)))[0]
    return Z

# Randomly sample 5 points using independent Gaussian noise
s = np.random.normal(0, .1, 5)
X = np.random.uniform(-1, 1, 5)
Y_obs = X**2+s

# Create 4-D data by mapping each x in D to Legendre polynomials
Z = mapL(X)

plt.xlim(-1, 1)
plt.ylim(-2, 2)

x = np.arange(-1, 1, 0.01)

# Plot data
plt.plot(X, Y_obs, 'ro')
# Plot target
f = np.power(x, 2)
plt.plot(x, f, color='red', label='target fxn')

# Hypothesis h(z) = w^Tz = w0 + w1*z1 + w2*z2 + w3*z3 + w4*z4
# w = (Z^T*Z + lambda*I)^{-1}Z^T*y
L = [0, 1E-5, 1E-2, 1E0]
for l in L:
    ZtZ = np.matmul(np.transpose(Z), Z)
    lam_I = l*np.identity(Z.shape[0])
    Z_ = np.linalg.pinv(np.add(ZtZ, lam_I))
    Wreg = np.matmul(Z_, np.matmul(np.transpose(Z), Y_obs))

    # Plot h(x)
    mapZ = mapL(x)
    hx = np.dot(mapZ, np.transpose(Wreg))
    if l == 0:
        c = 'blue'
    elif l == 1E-5:
        c = 'purple'
    elif l == 1E-2:
        c = 'orange'
    else:
        c = 'green'
    plt.plot(x, hx, color=c, label='h(x): ' + str(l))

    # Compute average squared error
    Eout = np.mean(np.power(np.subtract(np.dot(mapZ, Wreg), f), 2))
    print("Eout of " + str(l) + ": " + str(Eout))

plt.title("Regularized Regression without Cross Validation")
plt.legend(loc='best')
plt.show()
