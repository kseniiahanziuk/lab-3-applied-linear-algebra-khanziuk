import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
print(ratings_matrix)

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)
print(ratings_matrix)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(R_demeaned, k=3)


def visualization(matrix, title):
    print(f"\n{title}: {matrix}")
    fig = plt.figure()
    matrix = matrix[:20, :]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], color='purple')
    ax.set_title(title)
    plt.show()

V = Vt.T
visualization(U, "Users(U)")
visualization(V, "Films(V)")