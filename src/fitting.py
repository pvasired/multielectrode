import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
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
    
    w = params.reshape(-1, X.shape[-1])
    
    # Get predicted probability of spike using current parameters
    response_mat = 1 / (1 + np.exp(-X @ w.T))
    yPred = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    yPred[yPred == 0] = 0.000001

    # Calculate negative log likelihood
    NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred))     # negative log likelihood for logistic

    if method == 'l1':
         penalty = reg*np.linalg.norm(w.flatten(), ord=1)     # penalty term according to l1 regularization
    elif method == 'l2':
         penalty = reg*np.linalg.norm(w.flatten())    # penalty term according to l2 regularization
    else:
         penalty = 0

    if verbose:
        print(NLL, penalty)
    return(NLL + penalty)

def fsigmoid(X, w):
    return 1.0 / (1.0 + np.exp(-X @ w))

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

def enforce_3D_monotonicity(index, Xdata, ydata, k=2, percentile=0.6, num_points=100):
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

def get_w(m, X, y, zero_prob, nll_null, prev_initialization=None):
    if prev_initialization is not None:
        new_x0 = np.random.normal(size= (m - len(prev_initialization)) * X.shape[-1])
        x0 = np.concatenate((prev_initialization.ravel(), new_x0))
    else:
        x0 = np.random.normal(size= m * X.shape[-1])
    bounds = []
    z = 1 - (1 - zero_prob)**(1/m)
    for j in range(m):
        bounds += [(None, np.log(z/(1-z))), (None, None), (None, None), (None, None)]
        
    opt = minimize(negLL_hotspot, x0=x0, bounds=bounds,
                       args=(X, y, False, 'none', 0))
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)

def fit_triplet_surface(X_expt, probs, T, starting_m=2, max_sites=8, 
                        R2_thresh=0.02, zero_prob=0.01):
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)

    ybar = np.mean(y_bin)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), np.zeros(X_expt.shape[-1])))
    nll_null = negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 0)

    m = starting_m
    last_opt = get_w(m, X_bin, y_bin, zero_prob, nll_null)
    last_R2 = last_opt[2]
    m += 1

    print(last_opt)
    breakFlag = 0
    while m <= max_sites:
        new_opt = get_w(m, X_bin, y_bin, zero_prob, nll_null,
                        prev_initialization=last_opt[0])
        new_R2 = new_opt[2]

        print(new_opt)
        if new_R2 - last_R2 <= R2_thresh:
            breakFlag = 1
            break

        last_opt = new_opt
        last_R2 = new_R2
        m += 1

    if not breakFlag:
        raise RuntimeError('Failed to converge after max number of sites.')

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
