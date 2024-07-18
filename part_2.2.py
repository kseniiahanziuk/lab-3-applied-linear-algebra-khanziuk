import numpy as np
import scipy.sparse.linalg as svds
import pandas as pd

file = 'ratings.csv'
df = pd.read_csv(file)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=30, axis=1)
print(ratings_matrix)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
print(ratings_matrix_filled)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds.svds(R_demeaned, k=3)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

predicted_data = preds_df.mask(~ratings_matrix.isna())
print(predicted_data)


def recommend_movies(user_id, recommendations_number=10):
    user_row_number = user_id
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    recommendations = sorted_user_predictions.head(recommendations_number)
    movies_df = pd.read_csv('movies.csv')
    recommended_movies = movies_df[movies_df['movieId'].isin(recommendations.index)]
    return recommended_movies[['title', 'genres']]


user_id = 1
print(f"Recommended movies for {user_id}: \n", recommend_movies(user_id))
