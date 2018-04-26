import random
import math
import itertools

import numpy as np
from cvxpy import *


def _solve(M, omega):
    n_1 = M.shape[0]
    n_2 = M.shape[1]
    X_ = Semidef(n_1 + n_2)
    objective = Minimize(trace(X_))

    constraints = [(X_ == X_.T)]  # add symmetric constraint.
    for i, j in omega:
        constr = (X_[i, j + n_1] == M[i, j])
        constraints.append(constr)
    problem = Problem(objective, constraints)
    problem.solve(solver=CVXOPT)

    print("STATUS       :", problem.status)
    print("OPTIMAL VALUE:", problem.value)

    X0 = X_.value
    # check optimizer's solution is symmetric.
    print("|X0-X0.T|_F  :", np.linalg.norm(np.subtract(X0, X0.T), "fro"))
    return X_.value[:n_1, n_1:]


def _generate_omega(n_1, n_2, m):
    return random.sample([(i, j) for i in range(n_1) for j in range(n_2)], m)


def _get_mask_matrix(n_1, n_2, omega):
    """
    If we observed entry (i, j) of matrix M, the entry of mask matrix is 1,
    Otherwise 0.
    """
    mask = np.zeros((n_1, n_2), dtype=np.int8)
    for i, j in omega:
        mask[i, j] = 1
    return mask


def _get_abs_max_from_matrix(M):
    return np.max(np.absolute(M))


def main(n_1, n_2, r, m):
    print("#row of M    :", n_1)
    print("#column of M :", n_2)
    print("#sample      :", m)
    L = np.random.randn(n_1, r)
    R = np.random.randn(n_2, r)
    M = np.dot(L, R.T)
    M_abs_max = _get_abs_max_from_matrix(M)
    # print("RANK of M    :", np.linalg.matrix_rank(M))
    print("|M|_*        :", np.linalg.norm(M, "nuc"))
    M_rank = np.linalg.matrix_rank(M)
    print("RANK of M    :", M_rank)
    U, S, V_T = np.linalg.svd(M, full_matrices=False)
    print(S)

    omega = _generate_omega(n_1, n_2, m)
    mask = _get_mask_matrix(n_1, n_2, omega)

    # M_ is for training data removing the data.
    # This block should not affect the result.
    # Just for defensive programming.
    # M_ = np.random.uniform(-M_abs_max, M_abs_max, M.shape)
    M_ = M.copy()
    np.place(M_, 1 - mask, M_abs_max * M_abs_max)
    X = _solve(M_, omega)
    X_rank = np.linalg.matrix_rank(X)
    print("RANK of X    :", X_rank)
    print("|X|_*        :", np.linalg.norm(X, "nuc"))
    U, S, V_T = np.linalg.svd(X, full_matrices=False)
    print(S)

    E = np.subtract(M, X)
    E_train = E.copy()
    np.place(E_train, 1 - mask, 0)
    print('TRAIN RMSE   :', np.linalg.norm(E_train, "fro") / m)
    E_test = E.copy()
    np.place(E_test, mask, 0)
    print('TEST  RMSE   :', np.linalg.norm(E_test, "fro") / (n_1 * n_2 - m))

    frobenius_norm_ratio = (
        np.linalg.norm(X - M, "fro") / np.linalg.norm(M, "fro"))
    print("|X-M|_F/|M|_F:", frobenius_norm_ratio)  # the metric of paper.


if __name__ == '__main__':
    n = 20
    p = 0.8
    main(n, n, r=2, m=int(n * n * p))
