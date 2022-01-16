from collections import Counter
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH_TO_MOVIELENS_SMALL = './ml-latest-small'
PATH_TO_MOVIELENS_1M = './ml-1m'
ALL_GENRES = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
              'Sci-Fi', 'Thriller', 'War']   
# leaving out 'Film-Noir', 'Documentary', 'Musical', 'Western' bc not enough examples 

def load_movielens_ratings(movie_min_count=1, user_min_count=1, max_movies=None, max_users=None, dataset='small'):
    """
    Load MovieLens ratings, either the small dataset (for testing) or the full 1M dataset.
    """
    assert dataset in {'small', '1m'}
    if movie_min_count > 1 and user_min_count > 1:
        print('Warning: both movie and user min counts > 1; cannot ensure both constraints are perfectly met')    
    if dataset == 'small':
        fn = os.path.join(PATH_TO_MOVIELENS_SMALL, 'ratings.csv')
        print('Loading from %s...' % fn)
        df = pd.read_csv(fn)
    else:
        fn = os.path.join(PATH_TO_MOVIELENS_1M, 'ratings.dat')
        print('Loading from %s...' % fn)
        df = pd.read_csv(fn, sep='::', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
    print('Finished loading: found %d ratings' % len(df))
    movies = df['movieId'].values
    users = df['userId'].values
    ratings = df['rating'].values * 1.0  
    
    if movie_min_count > 1 or max_movies is not None:
        orig_num_movies = len(set(movies))
        movie_counts = Counter(movies).most_common()  # sorted by most to least common
        movies_to_keep = [m for m,c in movie_counts if c >= movie_min_count]
        print('Dropped %d movies with < %d ratings' % 
              (orig_num_movies - len(movies_to_keep), movie_min_count))
        if max_movies is not None:
            movies_to_keep = movies_to_keep[:max_movies]  # keep n most rated movies
        idx_to_keep_m = np.isin(movies, movies_to_keep)
    else:
        idx_to_keep_m = np.ones(len(movies)).astype(bool)
    
    if user_min_count > 1 or max_users is not None:
        orig_num_users = len(set(users))
        user_counts = Counter(users).most_common()
        users_to_keep = [u for u,c in user_counts if c >= user_min_count]
        print('Dropped %d users with < %d ratings' %
              (orig_num_users - len(users_to_keep), user_min_count))
        if max_users is not None:
            users_to_keep = users_to_keep[:max_users]
        idx_to_keep_u = np.isin(users, users_to_keep)
    else:
        idx_to_keep_u = np.ones(len(users)).astype(bool)
    
    idx_to_keep = idx_to_keep_m & idx_to_keep_u
    movies = movies[idx_to_keep]
    movie_reindexing = {m:i for i,(m,c) in enumerate(Counter(movies).most_common())}
    movies = [movie_reindexing[m] for m in movies]
    users = users[idx_to_keep]
    user_reindexing = {u:i for i,(u,c) in enumerate(Counter(users).most_common())}
    users = [user_reindexing[u] for u in users]
    ratings = ratings[idx_to_keep]
    mat = csr_matrix((ratings, (users, movies)), shape=(len(user_reindexing), len(movie_reindexing)))
    mat = mat.toarray()
    print('Created ratings matrix:', mat.shape)
    return mat, user_reindexing, movie_reindexing
    
def load_movie_data(reindexing=None, dataset='small'):
    """
    Load metadata about movies, like genre.
    """
    assert dataset in {'small', '1m'}
    if dataset == 'small':
        fn = os.path.join(PATH_TO_MOVIELENS_SMALL, 'movies.csv')
        print('Loading from %s...' % fn)
        df = pd.read_csv(fn)
    else:
        fn = os.path.join(PATH_TO_MOVIELENS_1M, 'movies.dat')
        print('Loading from %s...' % fn)
        df = pd.read_csv(fn, sep='::', header=None, names=['movieId', 'title', 'genres'])
    print('Finished loading: found %d movies' % len(df))
    if reindexing is not None:
        valid_orig_idx = set(reindexing.keys())  # only keep movies that have new index
        df = df[df['movieId'].isin(valid_orig_idx)]
        df['movieId'] = df['movieId'].map(lambda x:reindexing[x])  # replace MovieLens indexing
        df = df.set_index('movieId')
        df.sort_index(inplace=True)
        print('Kept %d movies' % len(df))
    return df

def get_movies_and_rating_for_user(user_idx, R, movie_df):
    """
    Returns the movies and ratings for a given user.
    """
    user_ratings = R[user_idx]
    nonzero_idx = np.nonzero(user_ratings)
    user_ratings = user_ratings[nonzero_idx]  # only get filled-out ratings
    user_movies = movie_df.iloc[nonzero_idx].copy()  # movies seen by this user
    user_movies['rating_by_u%d' % user_idx] = user_ratings
    return user_movies
    
def get_user_average_over_genres(user_idx, R, movie_df, missing_val=0):
    """
    Returns a user's average rating over genres.
    """
    user_movies = get_movies_and_rating_for_user(user_idx, R, movie_df)
    avg_rating_for_genre = []
    for genre in ALL_GENRES:
        genre_movies = user_movies[user_movies['genres'].str.contains(genre)]
        if len(genre_movies) > 0:
            avg_rating_for_genre.append(genre_movies['rating_by_u%d' % user_idx].mean())
        else:
            avg_rating_for_genre.append(missing_val)
    return avg_rating_for_genre
    
def compute_genre_coo_mat(movie_df, min_n=100):
    """
    Computes the co-occurence matrix for movie genres.
    """
    genres_to_consider = []
    for genre in ALL_GENRES:
        genre_count = np.sum(movie_df['genres'].str.contains(genre).values)
        if genre_count >= min_n:
            genres_to_consider.append(genre)
        else:
            print('Dropping genre %s because n=%d < min_n=%d' % (genre, genre_count, min_n))

    n_genres = len(genres_to_consider)
    genre2idx = dict(zip(genres_to_consider, range(n_genres)))
    coo_mat = np.zeros((n_genres, n_genres))
    for movie_genres in movie_df.genres:
        movie_genres = movie_genres.split('|')
        genre_idx = np.array([genre2idx[g] for g in movie_genres if g in genre2idx])
        if len(genre_idx) > 0:
            indicator_vec = np.zeros(n_genres)
            indicator_vec[genre_idx] = 1.0
            coo_mat[genre_idx] = coo_mat[genre_idx] + indicator_vec
    return coo_mat, genres_to_consider
