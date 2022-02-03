import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000     # data set size
d = 2       # 2 classes of data

# Generate random training data
X = np.random.uniform(-1, 1, size=(N, d+1))
X[:, 0] = 1

# Calculate weights vector
w = np.random.uniform(-1, 1, size=(d+1))

# Compute true labels for the training data
Y = np.sign(np.dot(X, w))
ind_pos = np.where(Y == 1)[0]   # positive examples
ind_neg = np.where(Y == -1)[0]  # negative examples
# If there are two few positive or negative examples, then repeat
# w = np.random... until you find good values for w

# Plot points
plt.clf()
plt.plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')    # red dot points
plt.plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')    # blue 'x' points


# Generate random target function: f(x) = w^Tx
m = w[1] / w[2]
intercept = w[0] / w[2]
line_x = np.linspace(-1, 1, 100)
# Plot target function
y = (-m*line_x) - intercept
plt.plot(line_x, y, label='target fxn f')

# Label axes
plt.ylabel('Y-coordinate data values')
plt.xlabel('X-coordinate data values')
plt.title('Linearly separable data')


# Perceptron algorithm
def perceptron(X, Y):
    # To be trained
    w_train = np.random.uniform(-1, 1, size=(d + 1))

    # Variable to store number of iterations until convergence
    count = 0

    while True:
        Y1 = np.sign(np.dot(X, w_train))

        # Lists to keep track of misclassified points
        misclassified_X = []
        misclassified_Y = []

        # Check for misclassified point
        for i in range(len(Y1)):
            if Y1[i] != Y[i]:
                misclassified_X.append(X[i, :])
                misclassified_Y.append(Y[i])

        if len(misclassified_X) == 0:
            return w_train, count

        # Choose random misclassified point
        num = random.randint(0, len(misclassified_X)-1)
        pt_x = misclassified_X[num]
        pt_y = misclassified_Y[num]

        # Update final hypothesis
        w_train += np.dot(pt_x, pt_y)
        count += 1


# Run perceptron algorithm
result = perceptron(X, Y)

# Generate final hypothesis g
g = result[0]
m1 = g[1] / g[2]
intercept1 = g[0] / g[2]
line_x1 = np.linspace(-1, 1, 100)

# Plot final hypothesis g
y1 = (-m*line_x1) - intercept1
plt.plot(line_x1, y1, label='final hypothesis g')

# Add legend
plt.legend(loc='upper right')

plt.show()

# Print number of iterations until convergence
print("Number of iterations until convergence: " + str(result[1]))

# PART G
# Uncomment to run part G!!!
"""
# Empty list to keep track of number of updates until convergence for each time perceptron is executed
numUpdates = []

# Repeat algorithm for 100 experiments
for i in range(100):
    # Generate random training data
    X = np.random.uniform(-1, 1, size=(N, d + 1))
    X[:, 0] = 1

    # Calculate weights vector
    w = np.random.uniform(-1, 1, size=(d + 1))

    # Compute true labels for the training data
    Y = np.sign(np.dot(X, w))
    ind_pos = np.where(Y == 1)[0]  # positive examples
    ind_neg = np.where(Y == -1)[0]  # negative examples
    # If there are two few positive or negative examples, then repeat
    # w = np.random... until you find good values for w

    # Generate random target function: f(x) = w^Tx
    m = w[1] / w[2]
    intercept = w[0] / w[2]
    line_x = np.linspace(-1, 1, 100)
    y = (-m * line_x) - intercept

    # Run perceptron algorithm
    result = perceptron(X, Y)

    updates = result[1]
    numUpdates.append(updates)

# Plot histogram
plt.hist(numUpdates, bins=10)
plt.show()
"""




