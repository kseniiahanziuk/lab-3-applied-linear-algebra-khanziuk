import numpy as np


def svd(matrix):
    AAT = np.dot(matrix, matrix.T)
    eigenvalues, eigenvectors = np.linalg.eig(AAT)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    U = eigenvectors[:, sorted_indices]

    singular_values = np.sqrt(eigenvalues)
    V = np.dot(matrix.T, U) / singular_values
    sigma = np.diag(singular_values)

    print("Original matrix: \n", matrix)
    print("\nU: \n", U)
    print("\nV.T: \n", V.T)
    print("\nSigma: \n", sigma)
    print("\nReconstructed matrix: \n", np.dot(U, np.dot(sigma, V.T)).round(1))


matrix_A = np.array([[3, 5], [-1, 3]])
svd(matrix_A)


