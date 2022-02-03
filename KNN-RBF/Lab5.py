import numpy as np
import matplotlib.pyplot as plt

features_train = np.loadtxt("/Users/kellyromer/Downloads/features.small.train")
features_test = np.loadtxt("/Users/kellyromer/Downloads/features.small.test")

# knn classifier
def knn(test, train, k):
    distances = []
    for j in range(len(train)):
        d1 = train[j][1] - test[1]
        d2 = train[j][2] - test[2]
        d = np.sqrt(np.power(d1, 2) + np.power(d2, 2))
        distances.append([d, j])
    distances.sort()
    votes = [0]*10
    for i in range(k):
        vote_ind = train[distances[i + 1][1]][0]
        votes[int(vote_ind)] += 1
    best_neighbor = np.argmax(votes)
    return best_neighbor

k_list = [1, 11, 21, 31]

for i in range(len(k_list)):
    # run knn on each k
    best_neighbors = []
    for j in range(len(features_test)):
        res = knn(features_test[j], features_train, k_list[i])
        best_neighbors.append(res)

    # calculate prediction error
    p_error = 0
    for j in range(len(best_neighbors)):
        if best_neighbors[j] == features_test[j][0]:
            p_error += 1
    p_error = p_error / len(best_neighbors)
    print("Prediction error for k = ", k_list[i], ": ", p_error)

    # plot predicted labels for various k values
    scatter = plt.scatter(features_test[:, 1], features_test[:, 2], c=best_neighbors)
    plt.legend(*scatter.legend_elements())
    plt.title('Predicted labels k = ' + str(k_list[i]))
    plt.savefig('Predicted labels k = ' + str(k_list[i]) + '.png')
    plt.clf()

# plot true labels
s = plt.scatter(features_test[:, 1], features_test[:, 2], c=features_test[:, 0], label='True labels')
plt.title('True Labels')
plt.legend(*s.legend_elements())
plt.savefig('True_labels.png')
plt.clf()

# lambda fxn for phi
phi = lambda z: np.exp(-.5 * (z**2))

# rbf classifier
def rbf(test, train, r):
    alphas = []
    for i in range(len(train)):
        d1 = train[i][1] - test[1]
        d2 = train[i][2] - test[2]
        d = np.sqrt(np.power(d1, 2) + np.power(d2, 2))
        alpha = phi(d/r)
        alphas.append((alpha, i))
    votes = [0]*10
    for i in range(len(train)):
        vote_ind = train[(alphas[i][1])][0]
        votes[int(vote_ind)] += alphas[i][0]
    vote_max = np.argmax(votes)
    return vote_max

r_list = [0.01, 0.05, 0.1, 0.5, 1]

for i in range(len(r_list)):
    # run rbf on each r
    best_vals = []
    for j in range(len(features_test)):
        res1 = rbf(features_test[j], features_train, r_list[i])
        best_vals.append(res1)

    # calculate prediction error
    p_error = 0
    for j in range(len(best_vals)):
        if best_vals[j] == features_test[j][0]:
            p_error += 1
    p_error = p_error / len(best_vals)
    print("Prediction error for r = ", r_list[i], ": ", p_error)

    # plot predicted labels for various r values
    scatter = plt.scatter(features_test[:, 1], features_test[:, 2], c=best_vals)
    plt.legend(*scatter.legend_elements())
    plt.title('Predicted labels r = ' + str(r_list[i]))
    plt.savefig('Predicted labels r = ' + str(r_list[i]) + '.png')
    plt.clf()


plt.show()
