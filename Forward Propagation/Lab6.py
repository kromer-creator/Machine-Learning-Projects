import numpy as np

# A: Randomly generate data
x1 = np.random.uniform(-1, 1, 100)
x2 = np.random.uniform(-1, 1, 100)
x = np.dstack((np.ones(x1.shape), x1, x2))[0]

# Hypotheses
w1 = np.transpose(np.array([0, 1, -1]))
w2 = np.transpose(np.array([0, 1, 1]))
h1 = np.sign(np.matmul(x, w1))
h2 = np.sign(np.matmul(x, w2))

# B: Compute true labels
def xor(x):
    y_list = []
    for i in range(len(x)):
        if (h1[i] > 0 and h2[i] < 0) or (h1[i] < 0 and h2[i] > 0):
            y_list.append(1)
        else:
            y_list.append(-1)
    return y_list
y_list = np.array(xor(x))
print('XOR Results: ', y_list)

# C: Write weight matrices
# W^1, W^2, W^3
W1 = np.array([w1, w2])
W2 = np.array([[-1.5, 1, -1], [-1.5, -1, 1]])
W3 = np.array([1.5, 1, 1])
w_list = [W1, W2, W3]
print('w list: ', w_list)

# Implement the forward propagation algorithm
def fp(input_x, theta):
    x = input_x
    for w in w_list:
        s = np.matmul(x, np.transpose(w))
        theta_s = theta(s)
        x = np.c_[(np.ones(input_x.shape[0]), theta_s)]
    return x[:, 1]

# D: Predict labels using forward propagation ... what is E_in?
h = fp(x, np.sign)
print('Forward Propagation using theta = sign: ', h)

e_in = np.mean(np.power(np.subtract(h, y_list), 2))
print('E-in sign: ', e_in)

# E: Repeat d using tan ... what is E_in?
h_tan = fp(x, np.tanh)
print('Forward Propagation using theta = tan: ', h_tan)

e_in_2 = np.mean(np.power(np.subtract(h_tan, y_list), 2))
print('E-in tan: ', e_in_2)
