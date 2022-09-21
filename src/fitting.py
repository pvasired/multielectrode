import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from itertools import chain, combinations
import copy

def convertToBinaryClassifier(probs, num_trials, amplitudes, degree=1, interaction=True):
    y = []
    X = []
    for j in range(len(amplitudes)):
        num1s = int(np.around(probs[j] * num_trials[j], 0))
        num0s = num_trials[j] - num1s

        X.append(np.tile(amplitudes[j], (num_trials[j], 1)))
        y.append(np.concatenate((np.ones(num1s), np.zeros(num0s))))
    
    if interaction == True:
        poly = PolynomialFeatures(degree)
        X = poly.fit_transform(np.concatenate(X))

    else:
        X = noInteractionPoly(np.concatenate(X), degree)
    
    y = np.concatenate(y)
    
    return X, y

def noInteractionPoly(amplitudes, degree):
    higher_order = []
    for i in range(degree):
        higher_order.append(amplitudes**(i+1))
    
    return sm.add_constant(np.hstack(higher_order))

def negLL(params, *args):
    X, y, verbose, method = args
    
    w = params
    
    # Get predicted probability of spike using current parameters
    yPred = 1 / (1 + np.exp(-X @ w))
    yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    yPred[yPred == 0] = 0.000001

    # Calculate negative log likelihood
    NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred))     # negative log likelihood for logistic

    if method == 'MAP':
         penalty = 0.5 * (w - mu) @ np.linalg.inv(cov) @ (w - mu)     # penalty term according to MAP regularization
    elif method == 'l1':
         penalty = l1_reg*np.linalg.norm(w, ord=1)     # penalty term according to l1 regularization
    elif method == 'l2':
         penalty = l2_reg*np.linalg.norm(w)    # penalty term according to l2 regularization
    else:
         penalty = 0

    if verbose:
        print(NLL, penalty)
    return(NLL + penalty)

def negLL_hotspot(params, *args):
    X, y, verbose, method, reg = args
    
    w = params.reshape(-1, X.shape[-1]).astype(float)

    prod = np.ones(len(X))
    for i in range(len(w)):
        prod *= (1 + np.exp(X @ w[i].T))
    prod -= 1

    prod[prod == 0] = 1e-5

    # yPred2 = 1 / (1 + np.exp(-np.log(prod)))

    # Get predicted probability of spike using current parameters
    # response_mat = 1 / (1 + np.exp(-X @ w.T))
    # yPred = 1 - np.multiply.reduce(1 - response_mat, axis=1)

    # print(np.sum(np.absolute(yPred - yPred2)))
    # yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    # yPred[yPred == 0] = 0.000001

    # Calculate negative log likelihood
    NLL2 = np.sum(np.log(1 + prod) - y * np.log(prod))
    # NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred))     # negative log likelihood for logistic

    # print(NLL, NLL2)

    if method == 'l1':
         penalty = reg*np.linalg.norm(w.flatten(), ord=1)     # penalty term according to l1 regularization
    elif method == 'l2':
         penalty = reg/2*np.linalg.norm(w.flatten())**2    # penalty term according to l2 regularization
    else:
         penalty = 0

    if verbose:
        print(NLL2, penalty)
    return(NLL2 + penalty)

def all_combos(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)    
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def negLL_hotspot_jac(params, *args):
    X, y, verbose, method, reg = args
    w = params.reshape(-1, X.shape[-1]).astype(float)
    
    prod = np.ones(len(X))
    for i in range(len(w)):
        prod = prod * (1 + np.exp(X @ w[i].T))
    prod = prod - 1

    prod[prod == 0] = 1e-5

    factors = np.zeros((len(w), len(X)))
    for i in range(len(w)):
        other_weights = np.setdiff1d(np.arange(len(w), dtype=int), i)
        other_combos = all_combos(other_weights)
        for j in range(len(other_combos)):
            other_combo = np.array(other_combos[j])
            if len(other_combo) > 0:
                factors[i] = factors[i] + np.exp(X @ np.sum(w[other_combo], axis=0).T)

    factors = factors + 1

    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        term1 = X.T @ (1 / (1 + np.exp(-X @ w[i].T)))
        term2 = -X.T @ (y * np.exp(X @ w[i].T) * factors[i] / prod)

        grad[i] = term1 + term2

    grad = grad.ravel()

    if method == 'l2':
         grad += reg * params    # penalty term according to l2 regularization

    return grad

def negLL_MMC(w, X):
    X = sm.add_constant(X, has_constant='add')

    prod = np.ones(len(X))
    for i in range(len(w)):
        prod *= (1 + np.exp(X @ w[i].T))
    prod -= 1

    yPred = prod / (prod + 1)

    factors = np.zeros((len(w), len(X)))
    for i in range(len(w)):
        other_weights = np.setdiff1d(np.arange(len(w), dtype=int), i)
        other_combos = all_combos(other_weights)
        for j in range(len(other_combos)):
            other_combo = np.array(other_combos[j])
            if len(other_combo) > 0:
                factors[i] = factors[i] + np.exp(X @ np.sum(w[other_combo], axis=0).T)

    factors = factors + 1

    MMC = np.zeros(len(X))
    for j in range(len(X)):

        grad0 = np.zeros_like(w, dtype=float)
        for i in range(len(w)):
            term1 = X[j] / (1 + np.exp(-X[j] @ w[i]))

            grad0[i] = term1

        grad1 = np.zeros_like(w, dtype=float)
        for i in range(len(w)):
            term1 = X[j] / (1 + np.exp(-X[j] @ w[i]))
            term2 = -X[j] * np.exp(X[j] @ w[i]) * factors[i][j] / prod[j]

            grad1[i] = term1 + term2

        MMC[j] = (1 - yPred[j]) * np.linalg.norm(grad0, 'fro') + yPred[j] * np.linalg.norm(grad1, 'fro')

    return MMC

def fsigmoid(X, w):
    return 1.0 / (1.0 + np.exp(-X @ w))

def clean_probs_triplet(probs,flip_thr=0.25,
                        zero_thr=0.15):

    if np.amax(probs) >= flip_thr:
        zero_inds = np.where(probs <= zero_thr)[0]
        '''
        Find the amplitude with probability closest to 0.5. Flip all 0s above 
        this amplitude to 1s, force all 0s below this amp to exactly 0
        '''
        thr_idx = np.argmin(np.absolute(probs - 0.5))
        zero_inds_1 = zero_inds[zero_inds > thr_idx]
        zero_inds_0 = zero_inds[zero_inds < thr_idx]

        probs_cleaned = probs.copy()
        probs_cleaned[zero_inds_1] = 1
        probs_cleaned[zero_inds_0] = 0

        return probs_cleaned

    return probs

def disambiguate_sigmoid(sigmoid_, spont_limit = 0.2, noise_limit = 0.0, thr_prob=0.5):
    sigmoid = copy.copy(sigmoid_)
    if np.max(sigmoid) <= spont_limit:
        return sigmoid
    above_limit = np.argwhere(sigmoid > spont_limit).flatten()
    
    i = np.argmin(np.abs(sigmoid[above_limit]-thr_prob))
    upper_tail = sigmoid[min(above_limit[i] + 1, len(sigmoid) - 1):]
    upper_tail[upper_tail<=noise_limit] = 1
    
    sigmoid[min(above_limit[i] + 1, len(sigmoid) - 1):] = upper_tail
    return sigmoid

def sigmoidND_nonlinear(X, w):
    response_mat = 1 / (1 + np.exp(-X @ w.T))
    response = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    return response

def sigmoidND_nonlinear_max(X, w):
    combos = all_combos(np.arange(len(w), dtype=int))

    max_val = np.ones(len(X)) * -np.inf
    for i in range(len(combos)):
        if len(np.array(combos[i])) > 0:
            subset = np.array(combos[i])
            subset_val = X @ np.sum(w[subset, :], axis=0)

            larger_inds = np.where(subset_val > max_val)[0]
            max_val[larger_inds] = subset_val[larger_inds]

    return 1 / (1 + np.exp(-max_val))

def Fisher_max(X, w, l2_reg=0):
    sigma = sigmoidND_nonlinear_max(X, w)

    F = np.zeros((w.shape[-1], w.shape[-1]))
    for i in range(len(sigma)):
        F += sigma[i] * (1 - sigma[i]) * np.outer(X[i], X[i])

    F = F / len(X)
    F += l2_reg * np.eye(w.shape[-1])

    return F

def var_max(X_l, X_u, w, l2_reg=0):
    F = Fisher_max(X_l, w, l2_reg=l2_reg)

    sigma_u = sigmoidND_nonlinear_max(X_u, w)
    var = np.zeros(len(X_u))

    for i in range(len(var)):
        c_i = sigma_u[i] * (1 - sigma_u[i]) * X_u[i]
        var[i] = c_i @ (np.linalg.inv(F) @ c_i) 

    return var

def enforce_3D_monotonicity(index, Xdata, ydata, k=2, percentile=0.9, num_points=100):
    point = Xdata[index]
    direction = point / np.linalg.norm(point)

    scaling = np.linspace(0.1, np.linalg.norm(point), num_points)
    closest_line = []
    for j in range(len(scaling)):
        curr = scaling[j] * direction
        dists = cdist(Xdata, curr[:, None].T).flatten()
        closest_inds = np.setdiff1d(np.argsort(dists)[:k], index)

        closest_line.append(closest_inds)

    line_inds = np.unique(np.concatenate(closest_line))

    if len(line_inds) > 0:
        if ydata[index] >= percentile * np.amax(ydata[line_inds]):
            return True
        
        else:
            return False
    
    else:
        return False

# def enforce_3D_monotonicity(index, Xdata, ydata, Tdata, k=2, percentile=0.9, num_points=100, mono_thr=0.5, noise_thr=0.2,
#                             norm_thr=1.3):

#     point = Xdata[index]
#     direction = point / np.linalg.norm(point)

#     scaling = np.linspace(0.1, np.linalg.norm(point), num_points)
#     closest_line = []
#     for j in range(len(scaling)):
#         curr = scaling[j] * direction
#         dists = cdist(Xdata, curr[:, None].T).flatten()
#         closest_inds = np.setdiff1d(np.argsort(dists)[:k], index)

#         closest_line.append(closest_inds)

#     line_inds = np.unique(np.concatenate(closest_line))

#     if len(line_inds) > 0:
#         if ydata[index] >= percentile * np.amax(ydata[line_inds]):
#             return Xdata[index], ydata[index], Tdata[index]
        
#         elif np.amax(ydata[line_inds]) >= mono_thr and ydata[index] <= noise_thr:
#             if np.linalg.norm(point) > norm_thr * np.linalg.norm(Xdata[line_inds[np.argmax(ydata[line_inds])]]):
#                 return Xdata[index], 1, Tdata[index]

#     elif ydata[index] > noise_thr:
#         return Xdata[index], ydata[index], Tdata[index]

# def triplet_disambiguation(Xdata, ydata, Tdata, dir_thr=0.1, spont_thr=0.4, noise_thr=0.2, mono_factor=0.6,
#                             norm_factor=1):

#     ymono = []
#     Xmono = []
#     Tmono = []
#     for index in range(len(Xdata)):
#         point = Xdata[index]
#         norm = np.linalg.norm(point)
#         direction = point / norm

#         other_inds = np.setdiff1d(np.arange(len(Xdata), dtype=int), index)
#         other_points = Xdata[other_inds]
#         other_norms = np.linalg.norm(other_points, axis=1)
#         other_directions = other_points / other_norms[:, None]

#         line_inds = other_inds[np.where((other_norms < norm) & 
#                                         (np.linalg.norm(other_directions - direction, axis=1) <= dir_thr))[0]]

#         if len(line_inds) > 0:
#             if ydata[index] >= mono_factor * np.amax(ydata[line_inds]):
#                 Xmono.append(Xdata[index])
#                 ymono.append(ydata[index])
#                 Tmono.append(Tdata[index])

#             elif np.amax(ydata[line_inds]) >= spont_thr and ydata[index] <= noise_thr:
#                 if norm > norm_factor * np.linalg.norm(Xdata[line_inds[np.argmax(ydata[line_inds])]]):
#                     Xmono.append(Xdata[index])
#                     ymono.append(1)
#                     Tmono.append(Tdata[index])

#             elif np.amax(ydata[line_inds]) <= noise_thr:
#                 Xmono.append(Xdata[index])
#                 ymono.append(ydata[index])
#                 Tmono.append(Tdata[index])

#     return np.array(Xmono), np.array(ymono), np.array(Tmono)

def get_w(m, X, y, nll_null, zero_prob=0.01, initialization=None, prev_initialization=None, method='L-BFGS-B', jac=None,
          reg_method='none', reg=0):
    
    bounds = []
    z = 1 - (1 - zero_prob)**(1/m)

    if prev_initialization is not None:
        new_x0 = np.random.normal(size= ((m - len(prev_initialization)), X.shape[-1]))
        new_x0[:, 0] = np.log(z/(1-z))
        init = (prev_initialization / prev_initialization[:, 0][:, None]) * np.log(z/(1-z))
        x0 = np.vstack((init, new_x0)).ravel()
    elif initialization is not None:
        x0 = initialization.ravel()
    else:
        x0 = np.random.normal(size=(m, X.shape[-1]))
        x0[:,0] = np.log(z/(1-z))
        x0 = x0.ravel()
        
    for j in range(m):
        bounds += [(None, np.log(z/(1-z))), (None, None), (None, None), (None, None)]

    opt = minimize(negLL_hotspot, x0=x0, bounds=bounds,
                       args=(X, y, False, reg_method, reg), method=method,
                        jac=jac)
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)

def fit_triplet_surface(X_expt, probs, T, starting_m=2, max_sites=8, 
                        R2_thresh=0.02, zero_prob=0.01, verbose=False,
                        method='L-BFGS-B', jac=None, initialization=None, reg_method='none', reg=0):
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)

    ybar = np.mean(y_bin)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), np.zeros(X_expt.shape[-1])))
    nll_null = negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 0)

    m = starting_m
    last_opt = get_w(m, X_bin, y_bin, nll_null, initialization=initialization, zero_prob=zero_prob, method=method, jac=jac,
                     reg_method=reg_method, reg=reg)
    last_R2 = last_opt[2]
    m += 1

    if verbose:
        print(last_opt)
    # breakFlag = 0
    while m <= max_sites:
        new_opt = get_w(m, X_bin, y_bin, nll_null, zero_prob=zero_prob,
                        prev_initialization=last_opt[0],
                        method=method,
                        jac=jac,
                        reg_method=reg_method,
                        reg=reg)
        new_R2 = new_opt[2]

        if verbose:
            print(new_opt)
        if new_R2 - last_R2 <= R2_thresh:
            # breakFlag = 1
            break

        last_opt = new_opt
        last_R2 = new_R2
        m += 1

    # if not breakFlag:
    #     raise RuntimeError('Failed to converge after max number of sites.')

    return last_opt[0]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
