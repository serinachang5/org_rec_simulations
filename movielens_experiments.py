import argparse 
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import time
from movielens import *

def make_prediction(mu, bu, bi, X, Y):
    """
    Given current collaborative filtering model parameters, predict ratings.
    """
    n_users = len(bu)
    n_items = len(bi)
    distances = cdist(X, Y) ** 2  # gets pairwise distances between all rows in X and all rows in Y
    assert distances.shape == (n_users, n_items)
    pred = np.ones((n_users, n_items)) * mu
    pred = pred + bi
    pred = (pred.T + bu).T
    pred -= distances
    return pred

def compute_loss(R, mu, bu, bi, X, Y, lmb, return_components=False):
    """
    Computes regularized loss on predicted ratings.
    """
    pred = make_prediction(mu, bu, bi, X, Y)
    observed = ~np.isnan(R)
    sse = np.sum((R[observed] - pred[observed]) ** 2)  # only sum over observed ratings
    distances = cdist(X, Y) ** 2  # gets pairwise distances between all rows in X and all rows in Y
    distances = distances + (bi ** 2)
    distances = (distances.T + (bu ** 2)).T
    reg_loss = lmb * np.sum(distances)  # sum over all user, item pairs
    if return_components:
        return sse + reg_loss, sse, reg_loss
    return sse + reg_loss

def fit_latent_factors(R, n_dims, lmb, init_lr=0.2, decay=0.05, max_iters=50, max_mse=0.7, min_lr=0.01):
    """
    Fit distance-based collaborative filtering model.
    """
    mu = np.nanmean(R)
    print('Mean rating = %.3f' % mu)
    n_users, n_items = R.shape
    bu = np.random.random(n_users)
    X = np.random.random((n_users, n_dims))
    bi = np.random.random(n_items)
    Y = np.random.random((n_items, n_dims))
    observed_indices = np.argwhere(~np.isnan(R))
    print('Found %d observed ratings' % len(observed_indices))
    order = np.arange(len(observed_indices))
    it = 0
    mse = max_mse + 1
    while mse > max_mse and it < max_iters:
        lr = max(init_lr * (1 / (1 + (decay * it))), min_lr)
        np.random.shuffle(order)
        num_clips = 0
        for idx in order:
            u, i = observed_indices[idx]
            vec_diff = X[u] - Y[i]
            true_rui = R[u, i]
            pred_rui = mu + bu[u] + bi[i] - np.sum(vec_diff ** 2)
            if pred_rui > 100 or pred_rui < -100:
                print('User', u, bu[u], X[u])
                print('Item', i, bi[i], Y[i])
                print('True rating = %s, predicted rating = %.3f' % (true_rui, pred_rui))
                raise Exception('Too large of a prediction')
            err = true_rui - pred_rui
            if err > 3 or err < -3:
                err = np.clip(err, -3, 3)
                num_clips += 1
            bu[u] += lr * (err - (lmb * bu[u]))
            bi[i] += lr * (err - (lmb * bi[i]))
            X[u] -= lr * vec_diff * (err + lmb)
            Y[u] += lr * vec_diff * (err + lmb)
        pred = make_prediction(mu, bu, bi, X, Y)
        mse = MSE(R, pred)
        print('Iter %d: MSE = %.3f (num clips = %d, learning rate = %.3f)' % (it, mse, num_clips, lr))
        it += 1
    return mu, bu, bi, X, Y
    
def fit_latent_factors_wo_movie_bias(R, n_dims, lmb=0.01, init_lr=0.2, decay=0.05, min_lr=0.01, n_iters=50):
    """
    Fit distance-based collaborative filtering model; does not use movie biases.
    """
    mu = np.nanmean(R)
    print('Mean rating = %.3f' % mu)
    n_users, n_items = R.shape
    bu = np.random.random(n_users)
    X = np.random.random((n_users, n_dims))
    Y = np.random.random((n_items, n_dims))
    observed_indices = np.argwhere(~np.isnan(R))
    print('Found %d observed ratings' % len(observed_indices))
    order = np.arange(len(observed_indices))
    total_losses = []
    sses = []
    reg_losses = []
    argmin = None
    ts = time.time()
    for it in range(n_iters):
        lr = max(init_lr * (1 / (1 + (decay * it))), min_lr)  # decay learning rate over time
        np.random.shuffle(order)  # shuffle order of ratings in each iteration
        num_clips = 0
        for idx in order:
            u, i = observed_indices[idx]
            vec_diff = X[u] - Y[i]
            true_rui = R[u, i]
            pred_rui = mu + bu[u] - np.sum(vec_diff ** 2)
            if pred_rui > 100 or pred_rui < -100:
                print('User', u, bu[u], X[u])
                print('Item', i, Y[i])
                print('True rating = %s, predicted rating = %.3f' % (true_rui, pred_rui))
                raise Exception('Too large of a prediction')
            err = true_rui - pred_rui
            if err > 3 or err < -3:
                err = np.clip(err, -3, 3)
                num_clips += 1
            # update parameters with gradient descent
            bu[u] += lr * (err - (lmb * bu[u]))  
            X[u] -= lr * vec_diff * (err + lmb)
            Y[i] += lr * vec_diff * (err + lmb)
        loss, sse, reg_loss = compute_loss(R, mu, bu, np.zeros(n_items), X, Y, lmb, return_components=True)
        total_losses.append(loss)
        sses.append(sse)
        reg_losses.append(reg_loss)
        if loss == min(total_losses):  # if this is the lowest we've seen so far
            argmin = (bu.copy(), X.copy(), Y.copy())
        print('Iter %d: L = %.2f, SSE = %.2f, reg loss = %.2f (num clips = %d, learning rate = %.3f)' % 
              (it, loss, sse, reg_loss, num_clips, lr))
    total_time = time.time() - ts
    print('Finished fitting -> time = %.3fs [%.3fs per iteration]' % (total_time, total_time / n_iters))
    return (mu, lmb), (bu, X, Y), argmin, (total_losses, sses, reg_losses)

def fit_and_save_embeddings(dataset, n_dims, n_iters, include_movie_bias=False, movie_min_count=20, user_min_count=1):
    """
    Fit distance-based collaborative filtering model and save resulting user/movie embeddings.
    """
    ts = time.time()
    R, user_mapping, movie_mapping = load_movielens_ratings(dataset=dataset, user_min_count=user_min_count, 
                                                            movie_min_count=movie_min_count)
    num_ratings_per_user = np.sum(R>0, axis=1)
    print('First 5 users:', num_ratings_per_user[:5])  # should be in descending order
    num_ratings_per_movie = np.sum(R>0, axis=0)
    print('First 5 movies:', num_ratings_per_movie[:5])
    print('Num ratings per user: min=%d, median=%d, mean=%d, max=%d' % (
        np.min(num_ratings_per_user), np.median(num_ratings_per_user), 
        np.mean(num_ratings_per_user), np.max(num_ratings_per_user)))
    print('Num ratings per movie: min=%d, median=%d, mean=%d, max=%d' % (
        np.min(num_ratings_per_movie), np.median(num_ratings_per_movie), 
        np.mean(num_ratings_per_movie), np.max(num_ratings_per_movie)))
    R[R == 0] = np.nan
    if include_movie_bias:
        results = fit_latent_factors(R, n_dims=n_dims, lmb=0.1, init_lr=0.2, max_iters=n_iters)
    else:
        results = fit_latent_factors_wo_movie_bias(R, n_dims=n_dims, n_iters=n_iters)
    
    fn = 'euclidean_embs_%s_%dD' % (dataset, n_dims)
    if include_movie_bias:
        fn += '_with_bi.pkl'
    else:
        fn += '.pkl'
    print('Saving results in %s' % fn)
    f = open(fn, 'wb')
    pickle.dump(results, f)
    f.close()
    print('Finished. Entire process took %.2fs' % (time.time() - ts))

def simulate_organic_model_with_alpha(X, Y, noise_cov, n_trials=30, shrinkage_alpha=None, verbosity=1):
    """
    Simulate organic model with alpha that controls shrinkage level.
    """
    n_users, n_dim = X.shape
    assert Y.shape[1] == n_dim
    n_movies = Y.shape[0]
    assert noise_cov.shape == (n_dim, n_dim)
    noise_mean = np.zeros(n_dim)
    user_choices_per_trial = np.zeros((n_users, n_trials))
    if shrinkage_alpha is not None:
        assert shrinkage_alpha > 0 and shrinkage_alpha <= 1
        print('Running simulation with shrinkage alpha =', shrinkage_alpha)
    for t in range(n_trials):
        if verbosity > 0 and t % verbosity == 0:
            print('trial', t)
        for u in range(n_users):  # need to go user by user bc each user has their own noisy sample of movies
            noisy_movies = Y + np.random.multivariate_normal(noise_mean, noise_cov, n_movies)
            if shrinkage_alpha is not None:
                noisy_mean = np.mean(noisy_movies, axis=0)
                noisy_movies = ((1 - shrinkage_alpha) * noisy_movies) + (shrinkage_alpha * noisy_mean)
            noisy_dists = cdist([X[u]], noisy_movies)
            assert noisy_dists.shape == (1, n_movies)
            user_choice = np.argmin(noisy_dists[0])  # index of movie chosen by user
            user_choices_per_trial[u, t] = user_choice
    return user_choices_per_trial

def simulate_recommender_model_with_alpha(X, Y, noise_cov, n_trials=30, shrinkage_alpha=None, verbosity=1):
    """
    Simulate recommender model with alpha that controls shrinkage level.
    """
    n_users, n_dim = X.shape
    assert Y.shape[1] == n_dim
    n_movies = Y.shape[0]
    assert noise_cov.shape == (n_dim, n_dim)
    noise_mean = np.zeros(n_dim)
    user_choices_per_trial = np.zeros((n_users, n_trials))
    if shrinkage_alpha is not None:
        assert shrinkage_alpha > 0 and shrinkage_alpha <= 1
        print('Running simulation with shrinkage alpha =', shrinkage_alpha)
    for t in range(n_trials):
        if verbosity > 0 and t % verbosity == 0:
            print('trial', t)
        noisy_users = X + np.random.multivariate_normal(noise_mean, noise_cov, n_users)
        if shrinkage_alpha is not None:
            noisy_mean = np.mean(noisy_users, axis=0)
            noisy_users = ((1 - shrinkage_alpha) * noisy_users) + (shrinkage_alpha * noisy_mean)
        noisy_dists = cdist(noisy_users, Y)
        assert noisy_dists.shape == (n_users, n_movies)
        user_choice = np.argmin(noisy_dists, axis=1)  # index of movie chosen by user
        user_choices_per_trial[:, t] = user_choice
    return user_choices_per_trial
    
def get_multivariate_variance(cov):
    '''
    Returns the log product of the eigenvalues of the covariance matrix.
    '''
    eigenvals, eigenvecs = np.linalg.eig(cov)
    log_prod = np.sum(np.log(eigenvals))
    return log_prod

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='which MovieLens dataset to use', choices=['small', '1m'])
    parser.add_argument('dimensions', type=int, help='number of dimensions')
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--include_movie_bias', type=bool, default=False)
    parser.add_argument('--movie_min_count', type=int, default=20)
    parser.add_argument('--user_min_count', type=int, default=1)
    args = parser.parse_args()
    fit_and_save_embeddings(args.dataset, args.dimensions, args.n_iters, 
                            args.include_movie_bias, args.movie_min_count, args.user_min_count)