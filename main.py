import numpy as np
from pprint import pprint


def cal_eigenvalues_and_eigenvectors(A):
    """
    :param A:  n x n Hermitian matrix
    :return:
    """
    eigenvalues, normed_eigenvectors = np.linalg.eig(A)
    # Below two steps are redounding for readability
    lmd = eigenvalues
    v = normed_eigenvectors
    return lmd, v


def cal_determinant(M):
    return np.linalg.det(M)


def check_lemma2():
    """
    lmd: short for lambda, i.e., eigenvalues.
         "lambda" is not a good choice in python so I use lmd instead
    v  : normed_eigenvectors
    :return:
    """
    n = np.random.randint(low=3, high=10)  # Dimension of a Hermitian matrix
    C = np.matrix(np.random.rand(n, n))              # Seed Matrix
    A = (C.getH() + C)                         # Construct Hermitian matrix
    pprint("Pick a {} x {} matrix".format(n, n))
    pprint(A)

    lmd, v = cal_eigenvalues_and_eigenvectors(A)
    pprint("Lambda Shape : {}".format(lmd.shape))
    pprint("V Shape: {}".format(v.shape))

    # Now pick a dimension: i
    i = np.random.randint(low=1, high=n)
    pprint("Pick one dimension to check : {}".format(i))

    # Now pick a dimension: j
    j = np.random.randint(low=0, high=n)
    pprint("Pick one dimension to delete : {}".format(j))

    # Now, let's compute left side of equation (2) in paper
    left = v[ j - 1, i - 1] ** 2
    for k in range(0, n):
        if k == i - 1:
            continue
        left *= (lmd[i - 1] - lmd[k])

    pprint("Left side equals to {}".format(left))

    # Now, let's compute right side of the equation (2) in paper

    right = 1
    M = np.delete(A, (j - 1), axis=0)
    M_j = np.delete(M, (j - 1), axis=1)
    lmd_M_j, v_M_j = cal_eigenvalues_and_eigenvectors(M_j)
    for k in range(0, n - 1):
        right *= (lmd[i - 1] - lmd_M_j[k])

    pprint("Right side equals to {}".format(right))

    assert np.abs(left - right) < 1e-5, "left side  {} does not equal to the right side {}.".format(left, right)


if __name__ == '__main__':
    check_lemma2()
