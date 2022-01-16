import numpy as np
import matplotlib.pyplot as plt

#######################################
# Results from theorems
#######################################
### E[y_ki]
def get_expected_yki(var_us, var_it, var_noi, x, mode, estimate='mle'):
    assert mode in {'org', 'rec'}
    assert estimate in {'mle', 'map'}
    if mode == 'org':
        if estimate == 'mle':
            return get_expected_yki_org_mle(var_us, var_it, var_noi, x)
        else:
            return get_expected_yki_org_map(var_us, var_it, var_noi, x)
    else:
        if estimate == 'mle':
            return get_expected_yki_rec_mle(var_us, var_it, var_noi, x)
        else:
            return get_expected_yki_rec_map(var_us, var_it, var_noi, x)

def get_expected_yki_org_mle(var_us, var_it, var_org, x):
    num = var_it * x
    denom = var_it + var_org
    return num / denom

def get_expected_yki_org_map(var_us, var_it, var_org, x):
    return x

def get_expected_yki_rec_mle(var_us, var_it, var_rec, x):
    return x

def get_expected_yki_rec_map(var_us, var_it, var_rec, x):
    num = var_us * x
    denom = var_us + var_rec
    return num / denom

### Var[y_ki]
def get_var_yki(var_us, var_it, var_noi, x, mode, estimate='mle'):
    assert mode in {'org', 'rec'}
    assert estimate in {'mle', 'map'}
    if mode == 'org':
        if estimate == 'mle':
            return get_var_yki_org_mle(var_us, var_it, var_noi, x)
        else:
            return get_var_yki_org_map(var_us, var_it, var_noi, x)
    else:
        if estimate == 'mle':
            return get_var_yki_rec_mle(var_us, var_it, var_noi, x)
        else:
            return get_var_yki_rec_map(var_us, var_it, var_noi, x)

def get_var_yki_org_mle(var_us, var_it, var_org, x):
    return 1 / ((1 / var_it) + (1 / var_org))

def get_var_yki_org_map(var_us, var_it, var_org, x):
    return 1 / ((1 / var_it) + (1 / var_org))

def get_var_yki_rec_mle(var_us, var_it, var_rec, x):
    return var_rec

def get_var_yki_rec_map(var_us, var_it, var_rec, x):
    num = (var_us ** 2) * var_rec
    denom = (var_us + var_rec) ** 2
    return num / denom
    
### MSE(x)
def get_mse_at_x(var_us, var_it, var_noi, x, mode, estimate='mle'):
    assert mode in {'org', 'rec'}
    assert estimate in {'mle', 'map'}
    bias = get_expected_yki(var_us, var_it, var_noi, x, mode, estimate) - x
    var = get_var_yki(var_us, var_it, var_noi, x, mode, estimate)
    mse = (bias ** 2) + var
    return mse

### Var[Y_k]
def get_expected_Yk_var(var_us, var_it, var_noi, mode, estimate='mle', m=1000):
    assert mode in {'org', 'rec'}
    assert estimate in {'mle', 'map'}
    if mode == 'org':
        if estimate == 'mle':
            return get_expected_Yk_var_org_mle(var_us, var_it, var_noi, m)
        else:
            return get_expected_Yk_var_org_map(var_us, var_it, var_noi, m)
    else:
        if estimate == 'mle':
            return get_expected_Yk_var_rec_mle(var_us, var_it, var_noi, m)
        else:
            return get_expected_Yk_var_rec_map(var_us, var_it, var_noi, m)
        
def get_expected_Yk_var_org_mle(var_us, var_it, var_org, m):
    var = ((var_it / (var_it + var_org)) ** 2) * var_us
    var = var + get_var_yki_org_mle(var_us, var_it, var_org, 0)
    scaling = (m - 1) / m
    return scaling * var

def get_expected_Yk_var_org_map(var_us, var_it, var_org, m):
    var = var_us + get_var_yki_org_mle(var_us, var_it, var_org, 0)
    scaling = (m - 1) / m
    return scaling * var
    
def get_expected_Yk_var_rec_mle(var_us, var_it, var_rec, m):
    var = var_us + var_rec
    scaling = (m - 1) / m
    return scaling * var

def get_expected_Yk_var_rec_map(var_us, var_it, var_rec, m):
    var = (var_us ** 2) / (var_us + var_rec)
    scaling = (m - 1) / m
    return scaling * var


#######################################
# Simulations
#######################################
def run_trial(var_us, var_it, var_noi, users, items, mode, estimate='mle',
              return_chosen_indices=False):
    assert mode in {'no_noise', 'org', 'rec'}
    if mode == 'no_noise':  # oracle: match every user to closest item
        assert estimate is None
        items_broadcasted = np.tile(items, len(users)).reshape((len(users), len(items)))  # row of items per user
        dists = np.absolute((items_broadcasted.T - users).T)  # dists_ij = true dist from user i to item j
    elif mode == 'org':
        assert estimate in {'mle', 'map'}
        user_estimates_of_items = np.random.normal(items, np.sqrt(var_noi), (len(users), len(items)))
        if estimate == 'map':
            scaling = var_it / (var_it + var_noi)
            user_estimates_of_items = scaling * user_estimates_of_items
        dists = np.absolute((user_estimates_of_items.T - users).T)  # dists_ij = dist from user i to i's estimate of item j
    else:
        assert estimate in {'mle', 'map'}
        system_estimates_of_users = np.random.normal(users, np.sqrt(var_noi))
        if estimate == 'map':
            scaling = var_us / (var_us + var_noi)
            system_estimates_of_users = scaling * system_estimates_of_users
        items_broadcasted = np.tile(items, len(users)).reshape((len(users), len(items)))  # row of items per user
        dists = np.absolute((items_broadcasted.T - system_estimates_of_users).T)  # dists_ij = dist from system's estimate of user i to item j
            
    chosen_indices = np.argmin(dists, axis=1) # minimum distance in each row
    chosen_items = items[chosen_indices]  # y_ki for each user 
    if return_chosen_indices:
        return chosen_indices, chosen_items
    return chosen_items

def sample_positions(dist, mean, var, n):
    assert dist in {'normal', 'uniform', 'bimodal', 'laplace'}
    if dist == 'normal':
        users = np.random.normal(mean, np.sqrt(var), n)
    elif dist == 'uniform':
        width = np.sqrt(12 * var)
        left = mean - (width/2)
        right = mean + (width/2)
        users = np.random.uniform(left, right, n)
    elif dist == 'bimodal':
        half_width = np.sqrt(var - 0.25)  # assume individual variances are 0.25
        users = np.random.normal(mean - half_width, 0.5, n)  # first assume all in class 1
        in_class_2 = np.random.uniform(0, 1, n) > 0.5
        class_2 = np.random.normal(mean + half_width, 0.5, np.sum(in_class_2))
        users[in_class_2] = class_2
    else:
        scale = np.sqrt(var / 2)
        users = np.random.laplace(mean, scale, n)
    return users
    
def plot_yki_over_n(var_us, var_it, var_noi, x_i, modes_and_estimates, n_range, num_trials_per_n, 
                    ax, colors, labels=None, ax_var=None, set_axis_labels=True, item_dist='normal', 
                    plot_expected=True, verbosity=20):
    assert all([mode in {'no_noise', 'org', 'rec'} for mode, est in modes_and_estimates])
    assert len(colors) == len(modes_and_estimates)
    users = np.array([x_i])
    mean_per_n = {(mode,est):[] for mode, est in modes_and_estimates}
    var_per_n = {(mode,est):[] for mode, est in modes_and_estimates}
    for n in n_range:
        if verbosity >= 1 and n % verbosity == 0:
            print(n)
        results_per_trial = {(mode, est):[] for mode, est in modes_and_estimates}
        for trial in range(num_trials_per_n):
            items = sample_positions(item_dist, 0, var_it, n)
            for mode, est in modes_and_estimates:
                chosen_item = run_trial(var_us, var_it, var_noi, users, items, mode, est)[0]
                results_per_trial[(mode, est)].append(chosen_item)
        
        for mode, est in modes_and_estimates:
            mean = np.mean(results_per_trial[(mode,est)])
            mean_per_n[(mode,est)].append(mean)
            var = np.var(results_per_trial[(mode,est)])
            var_per_n[(mode,est)].append(var)
    
    for i, (mode, est) in enumerate(modes_and_estimates):
        if labels is None:
            label = mode if mode == 'no_noise' else '%s-%s' % (mode, est)
        else:
            label = labels[i]
        color = colors[i]
        ax.plot(n_range, mean_per_n[(mode, est)], label=label, color=color, linewidth=2)
        if plot_expected and mode != 'no_noise':
            expected = get_expected_yki(var_us, var_it, var_noi, x_i, mode, est)
            ax.plot([min(n_range), max(n_range)], [expected, expected], label='%s expected' % label, 
                    color=color, linestyle='dashed', alpha=0.8)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=14)
    if set_axis_labels:
        ax.set_xlabel('Number of items available (n)', fontsize=14)
        ax.set_ylabel('Average match for user i', fontsize=14)
        ax.legend(fontsize=14)
    
    if ax_var is not None:
        for i, (mode, est) in enumerate(modes_and_estimates):
            if labels is None:
                label = mode if mode == 'no_noise' else '%s-%s' % (mode, est)
            else:
                label = labels[i]
            color = colors[i]
            ax_var.plot(n_range, var_per_n[(mode, est)], label=label, color=color, linewidth=2)
            if plot_expected and mode != 'no_noise':
                expected = get_expected_Yk_var(var_us, var_it, var_noi, x_i, mode, est)
                ax_var.plot([min(n_range), max(n_range)], [expected, expected], label='%s expected' % label, 
                        color=color, linestyle='dashed', alpha=0.8)
        ax_var.grid(alpha=0.2)
        ax_var.tick_params(labelsize=14)
        if set_axis_labels:
            ax_var.set_xlabel('Number of items available (n)', fontsize=14)
            ax_var.set_ylabel('Variance in matches for user i', fontsize=14)
            ax_var.legend(fontsize=14)

def plot_individual_metric_over_x(var_us, var_it, var_noi, users, metric, modes_and_estimates, n, num_trials,
                                  ax, colors, linestyles=None, labels=None, set_axis_labels=True, item_dist='normal', 
                                  plot_yx=True, verbose=True):
    assert metric in ['expected_yki', 'var_yki', 'mse']
    assert all([mode in {'no_noise', 'org', 'rec'} for mode, est in modes_and_estimates])
    assert len(colors) == len(modes_and_estimates)
    users = sorted(users)
    results_per_trial = {(mode, est):[] for mode, est in modes_and_estimates}
    for trial in range(num_trials):
        items = sample_positions(item_dist, 0, var_it, n)
        for mode, est in modes_and_estimates:
            chosen_items = run_trial(var_us, var_it, var_noi, users, items, mode, est)
            results_per_trial[(mode, est)].append(chosen_items)
    
    for i, (mode, est) in enumerate(modes_and_estimates):
        if labels is None:
            label = mode if mode == 'no_noise' else '%s-%s' % (mode, est)
        else:
            label = labels[i]
        if linestyles is None:
            linestyle = 'solid'
        else:
            linestyle = linestyles[i]
        if metric == 'expected_yki':
            results = np.mean(np.array(results_per_trial[(mode,est)]), axis=0)
        elif metric == 'var_yki':
            results = np.var(np.array(results_per_trial[(mode,est)]), axis=0)
        else:
            all_chosen_items = np.array(results_per_trial[(mode,est)])  # trial x user
            all_mse = (all_chosen_items - users) ** 2
            results = np.mean(all_mse, axis=0)
        ax.plot(users, results, label=label, color=colors[i], linestyle=linestyle, linewidth=2)
    if plot_yx:
        ax.plot(users, users, color='grey', alpha=0.8, label='y=x', linestyle='dashed')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=14)
    if set_axis_labels:
        ax.set_xlabel('x_i', fontsize=14)
        ax.set_ylabel('Average for user at x_i', fontsize=14)
        ax.legend(fontsize=14)

def plot_Yk_variance_over_n(var_us, var_it, var_noi, m, modes_and_estimates, n_range, num_trials_per_n,
                            ax, colors, labels=None, set_axis_labels=True, user_dist='normal', item_dist='normal', 
                            plot_expected=True, verbose=True):
    assert all([mode in {'no_noise', 'org', 'rec'} for mode, est in modes_and_estimates])
    assert len(colors) == len(modes_and_estimates)
    var_per_n = {(mode,est):[] for mode, est in modes_and_estimates}
    for n in n_range:
        if verbose and n % 20 == 0:
            print(n)
        var_per_trial = {(mode, est):[] for mode, est in modes_and_estimates}
        for trial in range(num_trials_per_n):
            users = sample_positions(user_dist, 0, var_us, m)
            items = sample_positions(item_dist, 0, var_it, n)
            for mode, est in modes_and_estimates:
                chosen_items = run_trial(var_us, var_it, var_noi, users, items, mode, est)
                var_per_trial[(mode, est)].append(np.var(chosen_items))        
        for mode, est in modes_and_estimates:
            mean_var = np.mean(var_per_trial[(mode,est)])
            var_per_n[(mode,est)].append(mean_var)
    
    for i, (mode, est) in enumerate(modes_and_estimates):
        if labels is None:
            label = mode if mode == 'no_noise' else '%s-%s' % (mode, est)
        else:
            label = labels[i]
        color = colors[i]
        ax.plot(n_range, var_per_n[(mode, est)], label=label, color=color, linewidth=2)
        if plot_expected and mode != 'no_noise':
            expected = get_expected_Yk_var(var_us, var_it, var_noi, mode, est, m)
            ax.plot([min(n_range), max(n_range)], [expected, expected], label='%s expected' % label, 
                    color=color, linestyle='dashed', alpha=0.8)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=14)
    if set_axis_labels:
        ax.set_xlabel('Number of items available (n)', fontsize=14)
        ax.set_ylabel('Average variance of chosen items', fontsize=14)
        ax.legend(fontsize=14)
    return var_per_n
        
def plot_yki_over_xi_and_n(var_mov, var_noi, X, n_ranges, T=1000, movie_dist='normal', movie_mean=0, plot_var=True):
    assert len(X) == len(n_ranges)
    if plot_var:
        fig, axes = plt.subplots(2, len(X), figsize=(15, 8))
    else:
        fig, axes = plt.subplots(1, len(X), figsize=(15, 5))
    for i, (x_i, n_range) in enumerate(zip(X, n_ranges)):
        print(x_i)
        if plot_var:
            plot_yki_over_n(var_mov, var_noi, x_i, n_range, axes[0][i], ax_var=axes[1][i], T=1000, set_axis_labels=False,
                            movie_dist=movie_dist, movie_mean=movie_mean)
        else:
            plot_yki_over_n(var_mov, var_noi, x_i, n_range, axes[i], T=1000, set_axis_labels=False,
                            movie_dist=movie_dist, movie_mean=movie_mean)
    
    if plot_var:
        row1_ymin = np.min([ax.get_ylim()[0] for ax in axes[0]])
        row1_ymax = np.max([ax.get_ylim()[1] for ax in axes[0]])
        for ax in axes[0]:
            ax.set_ylim((row1_ymin, row1_ymax))
            ax.tick_params(labelsize=12)

        row2_ymin = np.min([ax.get_ylim()[0] for ax in axes[1]])
        row2_ymax = np.max([ax.get_ylim()[1] for ax in axes[1]])
        for ax in axes[1]:
            ax.set_ylim((row2_ymin, row2_ymax))
            ax.tick_params(labelsize=12)
        
    plt.show()

def test_Yk_variance(var_user, var_mov, var_noi, m, n, T=1000, 
                     user_dist='normal', user_mean=0, movie_dist='normal', movie_mean=0):    
    org_var_Yk = []
    rec_var_Yk = []
    for t in range(T):
        users = sample_positions(user_dist, user_mean, var_user, m)
        movies = sample_positions(movie_dist, movie_mean, var_mov, n)
        org_Yk = run_trial('organic', users, movies, var_noi)
        org_var_Yk.append(np.var(org_Yk))
        rec_Yk = run_trial('recommender', users, movies, var_noi)
        rec_var_Yk.append(np.var(rec_Yk))
    return org_var_Yk, rec_var_Yk