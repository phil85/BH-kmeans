import numpy as np
from bh_kmeans import bh_kmeans
import matplotlib.pyplot as plt

# %% Create dataset
X = np.array([[2, 1],
              [2, 3],
              [3, 2],
              [4, 1],
              [4, 3],
              [6, 1],
              [6, 3],
              [7, 2],
              [8, 3],
              [8, 1],
              [1, 8],
              [2, 8],
              [2, 6],
              [3, 7],
              [4, 8],
              [5, 8],
              [6, 7],
              [7, 8],
              [7, 6],
              [8, 8]])

# %% Create constraint set

ml = [(1, 9), (11, 18)]
cl = [(4, 12), (8, 19)]

# %% Set control parameter for penalty
p = 1

# %% Apply BH-kmeans

labels = bh_kmeans(X, n_clusters=2, ml=ml, cl=cl, p=p, random_state=24)

# %% Visualize result

plt.figure(figsize=(5, 5), dpi=100)

# Plot members of each cluster
for label in np.unique(labels):
    plt.scatter(X[labels == label, 0], X[labels == label, 1])

# Plot must-link constraints
for (i, j) in ml:
    plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='green', alpha=0.5)

# Plot cannot-link constraints
for (i, j) in cl:
    plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='red', alpha=0.5)

plt.show()

centers = np.zeros((2, 2))
for i in range(2):
    centers[i, :] = X[labels == i, :].mean(axis=0)

