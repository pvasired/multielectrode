# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection as model_selection
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from itertools import chain, combinations
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf, erfinv

def negLL_hotspot(params, *args):
    """
    Compute the negative log likelihood for a logistic regression
    binary classification task assuming the hotpot model of activation.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method (str): regularization method. 'l1', 'l2', and
                                 'MAP' with multivariate Gaussian prior 
                                 are supported.
                   reg (float): regularization parameter
                                In the case of MAP, reg consists of 
                                (regmap, mu, cov)
                                where regmap is a constant scalar
                                      mu is the mean vector
                                      cov is the covariance matrix

    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, trials, verbose, method, reg = args
    
    w = params.reshape(-1, X.shape[-1]).astype(float)

    # Negative log-likelihood calculation for hotspot activation
    # prod = np.ones(len(X))
    # for i in range(len(w)):
    #     prod *= (1 + np.exp(X @ w[i].T))
    # prod -= 1

    # prod[prod < 1e-10] = 1e-10  # prevent divide by 0 errors

    ### Deprecated calculation (gives same results but slower) ###
    # yPred2 = 1 / (1 + np.exp(-np.log(prod)))

    # Get predicted probability of spike using current parameters
    response_mat = expit(X @ w.T)

    episilon = 1e-9
    yPred = np.clip(1 - np.multiply.reduce(1 - response_mat, axis=1), episilon, 1 - episilon)
    
    # negative log likelihood for logistic
    NLL = -np.sum(trials * (y * np.log(yPred) + (1 - y) * np.log(1 - yPred)))
    ###

    # Calculate negative log likelihood
    # NLL2 = np.sum(np.log(1 + prod) - y * np.log(prod))

    # Add the regularization penalty term if desired
    penalty = 0
    if reg > 0:
        if method == 'l1':
            # penalty term according to l1 regularization
            penalty = reg*np.linalg.norm(w.flatten(), ord=1)
        elif method == 'l2':
            # penalty term according to l2 regularization
            penalty = reg/2*np.linalg.norm(w.flatten())**2
        elif method == 'MAP':
            regmap, mu, cov = reg
            # penalty term according to MAP with Gaussian prior
            penalty = regmap * 0.5 * (params - mu) @ np.linalg.inv(cov) @ (params - mu)

    if verbose:
        print(NLL, penalty)
        
    return(NLL + penalty)

def all_combos(iterable):
    """
    Compute the 'powerset' of an iterable defined as:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    
    Parameters:
    iterable: An iterable list or np.ndarray

    Returns:
    powerset: The powerset, or another iterable consisting of all 
              combinations of elements from the input iterable.
    """
    s = list(iterable)    
    return list(chain.from_iterable(
                        combinations(s, r) for r in range(len(s)+1)))

def negLL_hotspot_jac(params, *args):
    """
    Manually computed jacobian of negative log likelihood function
    assuming a hotspot model of activation. Manual gradients greatly
    improve runtime.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method (str): regularization method. 'l2' 
                                 and MAP are supported.
                   reg (float): regularization parameter
                                In the case of MAP, reg is as above
                                in negLL_hotspot()

    Returns:
    grad (np.ndarray): jacobian of negative log likelihood, same shape
                       as params
    """
    X, y, verbose, method, reg = args
    w = params.reshape(-1, X.shape[-1]).astype(float)
    
    # Complicated manual jacobian calculation, was verified
    # to produce the same results as automated differentiation methods
    prod = np.ones(len(X))
    for i in range(len(w)):
        prod = prod * (1 + np.exp(X @ w[i].T))
    prod = prod - 1

    prod[prod < 1e-10] = 1e-10  # prevent divide by 0 errors

    factors = np.zeros((len(w), len(X)))
    for i in range(len(w)):
        other_weights = np.setdiff1d(np.arange(len(w), dtype=int), i)
        other_combos = all_combos(other_weights)
        for j in range(len(other_combos)):
            other_combo = np.array(other_combos[j])
            if len(other_combo) > 0:
                factors[i] = factors[i] + np.exp(X @ np.sum(
                                            w[other_combo], axis=0).T)

    factors = factors + 1

    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        term1 = X.T @ (1 / (1 + np.exp(-X @ w[i].T)))
        term2 = -X.T @ (y * np.exp(X @ w[i].T) * factors[i] / prod)

        grad[i] = term1 + term2

    grad = grad.ravel()

    # penalty term according to l2 regularization
    if method == 'l2':
        grad += reg * params

    # penalty term according to MAP
    elif method == 'MAP':
        regmap, mu, cov = reg
        grad += regmap  * (np.linalg.inv(cov) @ (params - mu)).flatten()

    return grad

def get_monotone_probs_and_amps(amplitudes,probs_,trials,n_amps_blank=0, st=0.5, return_inds=False):
    """
    A utility function that returns the set of amplitudes and probabilities
    that satisfy the monotone requirement.

    TODO: document.
    """
    probs = copy.deepcopy(probs_)

    # Zero out the first few amplitudes
    probs[0:n_amps_blank] = 0
    mono_inds = np.argwhere(enforce_noisy_monotonicity(probs, st=st)).flatten()

    if not return_inds:
        return amplitudes[mono_inds],probs[mono_inds], trials[mono_inds]
    else:
        return amplitudes[mono_inds],probs[mono_inds], trials[mono_inds], mono_inds

def enforce_noisy_monotonicity(probs, st=.5, noise_limit=.8):
    """
    Enforces monotonicity in the raw probability data. Finds indices that 
    violate monotonicity and excludes them in a final set for fitting.

    Code written by Jeff Brown.

    TODO: document.
    """
    J_array = []
    max_value = st
    trigger = False

    for i in range(len(probs)):

        if probs[i] >= max_value*noise_limit:
            max_value = probs[i]
            trigger = True
            J_array += [1]
        else:

            if not trigger:
                J_array += [1]
            else:
                J_array += [0]

    J_array = np.array(J_array).astype(np.int16)

    if J_array[0] == 1 and sum(J_array) == 1:
        J_array[-1] = 1

    return J_array

# Numpy version of activation_probs()
def sigmoidND_nonlinear(X, w):
    """
    N-dimensional nonlinear sigmoid computed according to multi-
    hotspot model.
    
    Parameters:
    X (np.ndarray): Input amplitudes
    w (np.ndarray): Weight vector matrix

    Returns:
    response (np.ndarray): Probabilities with same length as X
    """
    response_mat = expit(X @ w.T)
    response = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    return response

# Need to check modificiation of input arrays in this
def generate_input_list(all_probs_, amps_, trials_, w_inits_array, min_prob,
                        slope_bound=100, reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], zero_prob=0.01,
                        R2_thresh=0.05, opt_verbose=False, test_size=0.2):
    """
    Generate input list for multiprocessing fitting of sigmoids
    to an entire array.
    
    Parameters:
    all_probs (cells x patterns x amplitudes np.ndarray): Probabilities
    amps (patterns x amplitudes x stimElecs np.ndarray): Amplitudes
    trials (patterns x amplitudes np.ndarray): Trials
    w_inits_array (cells x patterns np.ndarray of objects): Initial guesses
                                                            of parameters
                                                            
    Returns:
    input_list (list): formatted list ready for multiprocessing
    """
    all_probs = copy.deepcopy(all_probs_)
    amps = copy.deepcopy(amps_)
    trials = copy.deepcopy(trials_)

    input_list = []
    for i in range(len(all_probs)):
        for j in range(len(all_probs[i])):
            probs = all_probs[i][j]
            T = trials[j]
            X = amps[j]

            good_T_inds = np.where(T > 0)[0]
            probs, X, T = copy.deepcopy(probs[good_T_inds]), copy.deepcopy(X[good_T_inds]), copy.deepcopy(T[good_T_inds])

            if len(probs[probs > min_prob]) == 0:
                probs = np.array([])
                X = np.array([])
                T = np.array([])

            input_list += [(X, probs, T, w_inits_array[i][j], R2_thresh, test_size, 
                            reg_method, reg, slope_bound, zero_prob, opt_verbose)]

    return input_list
    
def fit_surface_earlystop(X_expt, probs, T, w_inits_, 
                          R2_thresh=0.1, test_size=0.2,
                        reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], 
                        slope_bound=100, zero_prob=0.01,
                        opt_verbose=False, verbose=True,
                        method='L-BFGS-B', jac=None, random_state=None):
    """
    Fitting function for fitting surfaces to nonlinear data with multi-hotspot model.
    This function is primarily a wrapper for calling get_w() in the framework of 
    early stopping using the McFadden pseudo-R2 metric.

    Parameters:
    X_expt (N x d np.ndarray): Input amplitudes
    probs (N x 1 np.ndarray): Probabilities corresponding to the amplitudes
    T (N x 1 np.ndarray): Trials at each amplitude
    w_inits (list): List of initial guessses for each number of hotspots. Each element
                    in the list is a (m x (d + 1)) np.ndarray with m the number of 
                    hotspots. This list should be generated externally.
    R2_thresh (float): Threshold used for determining when to stop adding hotspots
    zero_prob (float): Value for what the probability should be forced to be below
                       at an amplitude of 0-vector
    verbose (bool): Increases verbosity
    method (string): Method for optimization according to constrained optimization
                     methods available in scipy.optimize.minimize
    jac (function): Jacobian function if manually calculated
    reg_method (string): Regularization method. 'l2' is supported
    reg (float): Regularization parameter value
    min_prob (float): Minimum probability that must be exceeded in the dataset for
                      fitting to occur and to not return the null parameters

    Returns:
    last_opt[0] (m x (d + 1) np.ndarray): The optimized set of parameters for the 
                                          optimized number of hotspots m using
                                          McFadden Pseudo-R2 and early stopping
    w_inits (list): The new initial guesses for each number of hotspots for the
                    next possible iteration of fitting
    """
    w_inits = copy.deepcopy(w_inits_)
    if len(probs) == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return (deg_opt, 0, -1), w_inits

    X_const = sm.add_constant(X_expt, has_constant='add')
    X_train, X_test, y_train, y_test, T_train, T_test = model_selection.train_test_split(X_const, probs, T,
                                                                                         test_size=test_size, random_state=random_state)

    test_R2s = np.zeros(len(w_inits))
    opts = []
    for i in range(len(w_inits)):
        if reg_method == 'MAP':
            opt = get_w(w_inits[i], X_train, y_train, T_train, zero_prob=zero_prob,
                                        method=method, 
                                        jac=jac, 
                                        reg_method=reg_method,
                                        reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                                        verbose=opt_verbose, 
                                        slope_bound=slope_bound)
        else:
            opt = get_w(w_inits[i], X_train, y_train, T_train,
                                                        zero_prob=zero_prob, 
                                                        method=method, 
                                                        jac=jac, 
                                                        reg_method=reg_method, 
                                                        reg=reg, 
                                                        verbose=opt_verbose,
                                                        slope_bound=slope_bound)
        test_fun = negLL_hotspot(opt[0], X_test, y_test, T_test, opt_verbose, reg_method, reg[0])

        # Compute the negative log likelihood of the null model which only
        # includes an intercept
        ybar_test = np.mean(y_test)
        beta_null_test = np.log(ybar_test / (1 - ybar_test))
        null_weights_test = np.concatenate((np.array([beta_null_test]), 
                                             np.zeros(X_expt.shape[-1])))
        nll_null_test = negLL_hotspot(null_weights_test, X_test, y_test, T_test, False, reg_method, reg[0])

        test_R2 = 1 - test_fun / nll_null_test
        if verbose:
            print(f'Number of sites: {len(w_inits[i])}, Test R2: {test_R2}')
        test_R2s[i] = test_R2
        opts.append(opt)

        if i > 0:
            if test_R2s[i-1] > 0 and (test_R2s[i] - test_R2s[i-1]) / test_R2s[i-1] <= R2_thresh:
                return opts[i-1], w_inits
            
    return opt, w_inits
    
def get_w(w_init, X, y, T, zero_prob=0.01, method='L-BFGS-B', jac=None,
          reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], slope_bound=100, bias_bound=None, verbose=False,
        #   options={'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxfun': 15000}):
          options={'maxiter': 200000, 'ftol': 1e-15, 'maxfun': 200000}):
    """
    Fitting function for fitting data with a specified number of hotspots
    
    Parameters:
    w_init (m x (d + 1) np.ndarray): Initial guesses on parameters for model
                                     with m hotspots
    X (N x (d + 1) np.ndarray): Binary classification input data with constant term
    y (N x 1 np.ndarray): Binary classification output data (0s or 1s)
    nll_null (float): The negative log likelihood for the null model to the data 
    zero_prob (float): The forced maximum probability at 0-vector
    method (string): Optimization method according to constrained optimization
                     methods available in scipy.optimize.minimize
    jac (function): Manual jacobian function
    reg_method (string): Regularization method, only 'none' is currently supported
    reg (float): Regularization parameter

    Returns:
    weights (m x (d + 1) np.ndarrray): Fitted weight vector
    opt.fun (float): Minimized value of negative log likelihood
    R2 (float): McFadden pseudo-R2 value
    """

    z = 1 - (1 - zero_prob)**(1/len(w_init))

    # Set up bounds for constrained optimization
    bounds = []
    for j in range(len(w_init)):
        bounds += [(bias_bound, np.log(z/(1-z)))]
        for i in range(X.shape[-1] - 1):
            bounds += [(-slope_bound, slope_bound)]

    ybar = np.mean(y)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), 
                                   np.zeros(X.shape[-1]-1)))
    nll_null = negLL_hotspot(null_weights, X, y, T, False, reg_method, reg[0])

    # Optimize the weight vector with MLE
    opt = minimize(negLL_hotspot, x0=w_init.ravel(), bounds=bounds,
                       args=(X, y, T, verbose, reg_method, reg[0]), method=method,
                        jac=jac, options=options)
    
    # print (X.shape, opt.nit, opt.nfev, opt.njev, (1 - opt.fun / nll_null))
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)