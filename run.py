import random

import numpy as np
from cvxpy import *


def solve(M, sampled_indices):
    n_1 = M.shape[0]
    n_2 = M.shape[1]
    X_ = Semidef(n_1 + n_2)
    objective = Minimize(trace(X_))

    constraints = []
    for i, j in sampled_indices:
        constr = (X_[i, j + n_1] == M[i, j])
        constraints.append(constr)
    problem = Problem(objective, constraints)
    problem.solve(solver=CVXOPT)

    return X_.value[:n_1, n_1:]


def main(n_1, n_2, r, m):
    L = np.random.randn(n_1, r)
    R = np.random.randn(n_2, r)
    M = np.dot(L, R.T)

    all_indices = []
    for i in range(n_1):
        for j in range(n_2):
            all_indices.append((i, j))
    sampled_indices = random.sample(all_indices, m)

    X = solve(M, sampled_indices)
    print("ERROR:", (np.linalg.norm(X - M, "fro") / np.linalg.norm(M, "fro")))


if __name__ == '__main__':
    n = 40
    main(n, n, r=2, m=640)
