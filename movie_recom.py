
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

ratings = pd.read_csv("ratings.csv")  # Contains user, movie, and ratings
movies = pd.read_csv("movies.csv")    # Contains movie titles and genres


print("Ratings Dataset:")
print(ratings.head())
print("Movies Dataset:")
print(movies.head())


print("Missing Values in Ratings Dataset:")
print(ratings.isnull().sum())
print("Missing Values in Movies Dataset:")
print(movies.isnull().sum())


plt.figure(figsize=(10, 5))
sns.countplot(x='rating', data=ratings)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()


data = pd.merge(ratings, movies, on='movieId')
print("Merged Dataset:")
print(data.head())


top_rated = data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)
print("Top-rated Movies:")
print(top_rated)


user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)


cosine_sim = cosine_similarity(user_movie_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)


def recommend_movies(user_id, num_recommendations=5):
    similar_users = cosine_sim_df[user_id].sort_values(ascending=False).index[1:]
    recommended_movies = pd.Series(dtype=float)

    for similar_user in similar_users:
        user_ratings = user_movie_matrix.loc[similar_user]
        recommended_movies = recommended_movies.add(user_ratings, fill_value=0)
        if len(recommended_movies) >= num_recommendations:
            break

  
    recommended_movies = recommended_movies.sort_values(ascending=False)
    return recommended_movies.head(num_recommendations)

def calculate_rmse():
    predicted_ratings = cosine_sim @ user_movie_matrix.values
    actual_ratings = user_movie_matrix.values
    mask = actual_ratings > 0

    mse = mean_squared_error(actual_ratings[mask], predicted_ratings[mask])
    return sqrt(mse)

rmse = calculate_rmse()
print(f"Model RMSE: {rmse}")

# Prediction: Recommend Movies for User
user_id = 1  
print(f"Recommendations for User {user_id}:")
recommendations = recommend_movies(user_id)
print(recommendations)
