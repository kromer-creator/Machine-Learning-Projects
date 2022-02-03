import numpy as np
import matplotlib.pyplot as plt

# Create multiple subplots
fig, axs = plt.subplots(2)

# Use y = sin(pi*x) to get D = {(x1,y1),(x2,y2)}
def f(inpt):
    return np.sin(np.pi * inpt)
# Obtain random sample dataset
def sample():
    X = np.random.uniform(-1, 1, 2)
    X1 = X[0]
    X2 = X[1]
    Y1 = f(X1)
    Y2 = f(X2)
    D = [(X1, Y1), (X2, Y2)]
    return D

# Fxn for A
def fA(x, X1, X2, Y1, Y2):
    return (Y1-Y2)/(X1-X2)*x + (X1*Y2-X2*Y1)/(X1-X2)

# Fxn for C
def fC():
    return (Y1+Y2)/2

A = []
C = []
x = np.arange(-1., 1., .01)

# Sample D 1000 times randomly
for i in range(1000):
    D = sample()
    X1 = D[0][0]
    Y1 = D[0][1]
    X2 = D[1][0]
    Y2 = D[1][1]

    # Evaluate/plot g_D(x) for part A
    line_A = fA(x, X1, X2, Y1, Y2)
    axs[0].plot(x, line_A, color='green')
    A.append(line_A)

    # Evaluate/plot g_D(x) for part C
    line_C = []
    for j in range(200):
        line_C.append(fC())
    axs[1].plot(x, line_C, color='green')
    C.append(line_C)

# g_bar
g_bar_A = np.mean(A, axis=0)
print("g_bar A: " + str(np.mean(g_bar_A)))                  # should be near 0
g_bar_C = np.mean(C, axis=0)
print("g_bar C: " + str(np.mean(g_bar_C)))                  # should be near 0
axs[0].plot(x, g_bar_A, color='red', label='g_bar A')
axs[1].plot(x, g_bar_C, color='red', label='g_bar C')

# var(x)
var_A = np.mean(np.power(np.subtract(A, g_bar_A), 2), axis=0)    # should be near 1.69
print("Variance of A: " + str(np.mean(var_A)))
var_C = np.mean(np.power(np.subtract(C, g_bar_C), 2), axis=0)     # should be near 0.25
print("Variance of C: " + str(np.mean(var_C)))
axs[0].plot(x, var_A, color='blue', label='var(x) A')
axs[1].plot(x, var_C, color='blue', label='var(x) C')

# bias(x)
bias_A = np.power(np.subtract(g_bar_A, f(x)), 2)
print("Bias of A: " + str(np.mean(bias_A)))                     # should be near .21
bias_C = np.power(np.subtract(g_bar_C, f(x)), 2)
print("Bias of C: " + str(np.mean(bias_C)))                     # should be near .5
axs[0].plot(x, bias_A, color='orange', label='bias(x) A')
axs[1].plot(x, bias_C, color='orange', label='bias(x) C')

# Mean Eout
E_out_A = np.add(var_A, bias_A)
print("Eout of A: " + str(np.mean(E_out_A)))        # should be near 1.9
E_out_C = np.add(var_C, bias_C)
print("Eout of C: " + str(np.mean(E_out_C)))        # should be near .75
axs[0].plot(x, E_out_A, color='purple', label='Eout A')
axs[1].plot(x, E_out_C, color='purple', label='Eout C')

# Plot legend
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')

plt.show()
