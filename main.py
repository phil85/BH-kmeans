import numpy as np
# from blpkm_cc import blpkm_cc
from blpkm_cc_pulp import blpkm_cc
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

# %% Apply BLPKM-CC

labels = blpkm_cc(X, n_clusters=2, ml=ml, cl=cl, random_state=24)

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
    centers[i, :] = X[labels==i, :].mean(axis=0)


# """
# A set partitioning model of a wedding seating problem
#
# Authors: Stuart Mitchell 2009
# """
#
# import pulp
#
# max_tables = 5
# max_table_size = 4
# guests = "A B C D E F G I J K L M N O P Q R".split()
#
#
# def happiness(table):
#     """
#     Find the happiness of the table
#     - by calculating the maximum distance between the letters
#     """
#     return abs(ord(table[0]) - ord(table[-1]))
#
#
# # create list of all possible tables
# possible_tables = [tuple(c) for c in pulp.allcombinations(guests, max_table_size)]
#
# # create a binary variable to state that a table setting is used
# x = pulp.LpVariable.dicts(
#     "table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger
# )
#
# seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)
#
# seating_model += pulp.lpSum([happiness(table) * x[table] for table in possible_tables])
#
# # specify the maximum number of tables
# seating_model += (
#     pulp.lpSum([x[table] for table in possible_tables]) <= max_tables,
#     "Maximum_number_of_tables",
# )
#
# # A guest must seated at one and only one table
# for guest in guests:
#     seating_model += (
#         pulp.lpSum([x[table] for table in possible_tables if guest in table]) == 1,
#         "Must_seat_%s" % guest,
#     )
#
# seating_model.solve()
#
# print("The choosen tables are out of a total of %s:" % len(possible_tables))
# for table in possible_tables:
#     if x[table].value() == 1.0:
#         print(table)