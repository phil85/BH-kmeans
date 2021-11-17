from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
import gurobipy as gb
import numpy as np


def update_centers(X, centers, n_clusters, labels):
    """Update positions of cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (np.array): current cluster assignments of objects

    Returns:
        np.array: the updated positions of cluster centers

    """
    for i in range(n_clusters):
        centers[i] = X[labels == i, :].mean(axis=0)
    return centers


def assign_objects(X, centers, ml, cl, p):
    """Assigns objects to clusters

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples
        p (float): control parameter for penalty

    Returns:
        np.array: cluster labels for objects

    """

    # Compute model input
    n = X.shape[0]
    k = centers.shape[0]
    distances = cdist(X, centers)
    M = distances.max()
    assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

    # Create model
    m = gb.Model()

    # Add binary decision variables
    x = m.addVars(assignments, vtype=gb.GRB.BINARY)
    y = m.addVars(cl)
    z = m.addVars(ml)

    # Add objective function
    term1 = gb.quicksum(distances[i, j] * x[i, j] for i in range(n) for j in range(k))
    term2 = gb.quicksum(M * p * y[i, i_] for i, i_ in cl)
    term3 = gb.quicksum(M * p * z[i, i_] for i, i_ in ml)
    m.setObjective(term1 + term2 + term3)

    # Add constraints
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(x.sum('*', j) >= 1 for j in range(k))
    m.addConstrs(x[i, j] + x[i_, j] <= 1 + y[i, i_] for j in range(k) for i, i_ in cl)
    m.addConstrs(x[i, j] - x[i_, j] <= z[i, i_] for j in range(k) for i, i_ in ml)

    # Determine optimal solution
    m.optimize()

    # Get labels from optimal assignment
    labels = np.array([j for i, j in x.keys() if x[i, j].X > 0.5])

    return labels


def get_total_distance(X, centers, labels):
    """Computes total distance between objects and cluster centers

    Args:
        X (np.array): feature vectors of objects
        centers (np.array): current positions of cluster centers
        labels (np.array): current cluster assignments of objects

    Returns:
        float: total distance

    """
    dist = np.sqrt(((X - centers[labels, :]) ** 2).sum(axis=1)).sum()
    return dist


def blpkm_cc_soft(X, n_clusters, ml=[], cl=[], p=1, random_state=None, max_iter=100):
    """Finds partition of X subject to must-link and cannot-link constraints

    Args:
        X (np.array): feature vectors of objects
        n_clusters (int): predefined number of clusters
        ml (list): must-link pairs as a list of tuples
        cl (list): cannot-link pairs as a list of tuples
        p (float): control parameter for penalty
        random_state (int, RandomState instance): random state
        max_iter (int): maximum number of iterations of blpkm_cc algorithm

    Returns:
        np.array: cluster labels of objects

    """

    # Choose initial cluster centers randomly
    centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)

    # Assign objects
    labels = assign_objects(X, centers, ml, cl, p)

    # Initialize best labels
    best_labels = labels

    # Update centers
    centers = update_centers(X, centers, n_clusters, labels)

    # Compute total distance
    best_total_distance = get_total_distance(X, centers, labels)

    n_iter = 0
    while n_iter < max_iter:

        # Assign objects
        labels = assign_objects(X, centers, ml, cl, p)

        # Update centers
        centers = update_centers(X, centers, n_clusters, labels)

        # Compute total distance
        total_distance = get_total_distance(X, centers, labels)

        # Check stopping criterion
        if total_distance >= best_total_distance:
            break
        else:
            # Update best labels and best total distance
            best_labels = labels
            best_total_distance = total_distance

        # Increase iteration counter
        n_iter += 1

    return best_labels
