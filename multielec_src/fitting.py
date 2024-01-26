# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection as model_selection
import statsmodels.api as sm
from scipy.optimize import minimize
from itertools import chain, combinations
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convertToBinaryClassifier(probs_, num_trials_, amplitudes_, degree=1, 
                              interaction=True, min_trials=1, vote_by_majority=False):
    """
    Converts input g-sort data of probabilities, trials, and 
    amplitudes. Includes a functionality for converting to data to a
    polynomial transformation which is now largely deprecated.

    Parameters:
    probs (N x 1 np.ndarray): The input probabilities
    num_trials (N x 1 np.ndarray): The input number of trials used for
                                   each probability
    amplitudes (N x k np.ndarray): The amplitude (vectors) of current
                                   stimulus. Supports multi-electrode
                                   stimulation.
    
    Optional Arguments:
    degree (int): The polynomial transformation degree, default 1
    interaction (bool): Whether or not to include cross terms in the
                        construction of the polynomial transform.
    
    Return:
    X (np.ndarray): The binary classifier inputs with constant term
                    and possibly polynomial tranformation terms added.
                    The shape of this array is related to the number
                    of trials per amplitude, the number of amplitudes,
                    the stimulation current vector sizes, and the 
                    polynomical transformation degree.

    y (np.ndarray): The binary classifier outputs consisting of 0s and
                    1s, with the same length as X.
    """
    probs = copy.deepcopy(probs_)
    num_trials = copy.deepcopy(num_trials_)
    amplitudes = copy.deepcopy(amplitudes_)

    y = []
    X = []
    num_trials = num_trials.astype(int) # convert to integer
    for j in range(len(amplitudes)):
        if num_trials[j] >= min_trials:
            if vote_by_majority:
                num_trials[j] = 1
                probs[j] = np.around(probs[j], 0)

            # Calculate the number of 1s and 0s for each probability
            num1s = int(np.around(probs[j] * num_trials[j], 0))
            num0s = num_trials[j] - num1s

            # Append all the amplitudes for this probability
            X.append(np.tile(amplitudes[j], (num_trials[j], 1)))

            # Append the 0s and 1s for this probability
            y.append(np.concatenate((np.ones(num1s), np.zeros(num0s))))

    assert len(X) > 0, "No data points were found with enough trials"
    
    # If desired, perform a polynomial transformation with cross terms
    if interaction == True:
        poly = PolynomialFeatures(degree)
        X = poly.fit_transform(np.concatenate(X))

    # If no cross terms are desired
    else:
        X = noInteractionPoly(np.concatenate(X), degree)
    
    y = np.concatenate(y)
    
    return X, y

def noInteractionPoly(amplitudes, degree):
    """
    Constructs a non-interacting polynomial transformation with no
    cross terms included.

    Parameters:
    amplitudes (np.ndarray): Raw amplitude vectors
    degree (int): Degree of polynomial transformation


    Returns:
    X (np.ndarray): Same length as amplitudes, but expanded along 
                    axis 1 according to the degree of the transform
    """
    # For each degree, add the next non-interacting polynomial
    higher_order = []
    for i in range(degree):
        higher_order.append(amplitudes**(i+1))
    
    # Add a constant column to the output
    return sm.add_constant(np.hstack(higher_order), has_constant='add')

def negLL(params, *args):
    """
    Compute the negative log likelihood for a logistic regression
    binary classification task.

    Parameters:
    params (np.ndarray): Weight vector to be fit, same dimension as
                         axis=1 dimension of X (see below)
    *args (tuple): X (np.ndarray) output of convertToBinaryClassifier,
                   y (np.ndarray) output of convertToBinaryClassifier,
                   verbose (bool) increases verbosity
                   method: regularization method. 'MAP' (maximum a 
                           posteriori), 'l1', and 'l2' are supported.
    
    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, verbose, method = args
    
    w = params
    
    # Get predicted probability of spike using current parameters
    yPred = 1 / (1 + np.exp(-X @ w))
    yPred[yPred == 1] = 0.999999     # some errors when yPred is 
                                     # exactly 1 due to taking 
                                     # log(1 - 1)
    yPred[yPred == 0] = 0.000001

    # Calculate negative log likelihood
    NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred))     

    if method == 'MAP':
        # penalty term according to MAP regularization
        penalty = 0.5 * (w - mu) @ np.linalg.inv(cov) @ (w - mu)
    elif method == 'l1':
        # penalty term according to l1 regularization
        penalty = l1_reg*np.linalg.norm(w, ord=1)
    elif method == 'l2':
        # penalty term according to l2 regularization
        penalty = l2_reg*np.linalg.norm(w)
    else:
         penalty = 0

    if verbose:
        print(NLL, penalty)

    return(NLL + penalty)

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
    X, y, verbose, method, reg = args
    
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
    response_mat = 1 / (1 + np.exp(-X @ w.T))
    yPred = 1 - np.multiply.reduce(1 - response_mat, axis=1)

    # print(np.sum(np.absolute(yPred - yPred2)))
    yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    yPred[yPred == 0] = 0.000001
    
    # negative log likelihood for logistic
    NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred)) 
    ###

    # Calculate negative log likelihood
    # NLL2 = np.sum(np.log(1 + prod) - y * np.log(prod))

    # Add the regularization penalty term if desired
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
    else:
        penalty = 0

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
    response_mat = 1 / (1 + np.exp(-X @ w.T))
    response = 1 - np.multiply.reduce(1 - response_mat, axis=1)
    return response

# Need to check modificiation of input arrays in this
def generate_input_list(all_probs_, amps_, trials_, w_inits_array, min_prob,
                        bootstrapping=None, X_all=None,
                        slope_bound=20, reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], zero_prob=0.01,
                        R2_thresh=0.05, opt_verbose=False):
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

            if X_all is not None:
                input_list += [(X, probs, T, w_inits_array[i][j], bootstrapping, X_all[j],
                                reg_method, reg, slope_bound, zero_prob, R2_thresh, opt_verbose)]
            else:
                input_list += [(X, probs, T, w_inits_array[i][j], bootstrapping, None,
                                reg_method, reg, slope_bound, zero_prob, R2_thresh, opt_verbose)]

    return input_list

def selectivity_triplet(ws, targets, curr_min=-1.8, curr_max=1.8, num_currs=40):
    I1 = np.linspace(curr_min, curr_max, num_currs)
    I2 = np.linspace(curr_min, curr_max, num_currs)
    I3 = np.linspace(curr_min, curr_max, num_currs)

    X_triplet = sm.add_constant(np.array(np.meshgrid(I1, I2, I3)).T.reshape(-1,3), has_constant='add')
    X_1elec = []
    for i in range(3):
        X_elec = np.zeros((len(I1), 3))
        X_elec[:, i] = I1
        X_1elec.append(X_elec)

    X_1elec = sm.add_constant(np.vstack(X_1elec), has_constant='add')

    selec_product_triplet = np.ones(len(X_triplet))
    for i in range(len(ws)):
        if i in targets:
            selec_product_triplet = selec_product_triplet * sigmoidND_nonlinear(X_triplet, ws[i])
        else:
            selec_product_triplet = selec_product_triplet * (1 - sigmoidND_nonlinear(X_triplet, ws[i]))

    selec_product_1elec = np.ones(len(X_1elec))
    for i in range(len(ws)):
        if i in targets:
            selec_product_1elec = selec_product_1elec * sigmoidND_nonlinear(X_1elec, ws[i])
        else:
            selec_product_1elec = selec_product_1elec * (1 - sigmoidND_nonlinear(X_1elec, ws[i]))

    return np.amax(selec_product_triplet), np.amax(selec_product_1elec)

def fit_surface(X_expt, probs, T, w_inits_, bootstrapping=None, X_all=None,
                        reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], 
                        slope_bound=20, zero_prob=0.01,
                        R2_thresh=0.1, opt_verbose=False, verbose=False,
                        method='L-BFGS-B', jac=None):
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

    # Convert the data to binary classification data
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)

    # If the probability never gets large enough, return the degenerate parameters
    # The degenerate parameters are a bias term of -np.inf and all slopes set to 0
    # These parameters cause probs_pred to be an array of all 0s
    
    # If a large enough probability was detected, begin fitting

    # Compute the negative log likelihood of the null model which only
    # includes an intercept
    ybar = np.mean(y_bin)
    if ybar == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return (deg_opt, 0, -1), w_inits

    X_orig, y_orig = convertToBinaryClassifier(probs, T, X_expt, min_trials=1)

    # Now begin the McFadden pseudo-R2 early stopping loop
    if reg_method == 'MAP':
        last_opt = get_w(w_inits[0], X_bin, y_bin, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, 
                        reg=(reg[0], reg[1][0][0], reg[1][0][1]),
                        verbose=opt_verbose, slope_bound=slope_bound)
    else:
        last_opt = get_w(w_inits[0], X_bin, y_bin, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, reg=reg, verbose=opt_verbose,
                        slope_bound=slope_bound,
                        X_orig=X_orig, y_orig=y_orig)
    w_inits[0] = last_opt[0]
    last_R2 = last_opt[2]   # store the pseudo-R2 value for early stopping
                            # procedure
    BIC = len(w_inits[0].flatten()) * np.log(len(X_bin)) + 2 * last_opt[1]
    HQC = 2 * len(w_inits[0].flatten()) * np.log(np.log(len(X_bin))) + 2 * last_opt[1]
    if verbose:
        print(last_opt, last_R2, BIC, HQC)

    for i in range(1, len(w_inits)):
        # Refit with next number of sites
        if reg_method == 'MAP':
            new_opt = get_w(w_inits[i], X_bin, y_bin, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                            verbose=opt_verbose,
                            slope_bound=slope_bound)
        else:
            new_opt = get_w(w_inits[i], X_bin, y_bin, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=reg,
                            verbose=opt_verbose,
                            slope_bound=slope_bound,
                            X_orig=X_orig, y_orig=y_orig)
        w_inits[i] = new_opt[0]
        new_R2 = new_opt[2]
        BIC = len(w_inits[i].flatten()) * np.log(len(X_bin)) + 2 * new_opt[1]
        HQC = 2 * len(w_inits[i].flatten()) * np.log(np.log(len(X_bin))) + 2 * new_opt[1]

        if verbose:
            print(new_opt, new_R2, BIC, HQC)

        # If the pseudo-R2 improvement was too small, break and stop adding sites
        if last_R2 > 0 and (new_R2 - last_R2) / last_R2 <= R2_thresh:
            final_ind = i - 1
            break

        last_opt = new_opt
        last_R2 = new_R2
        final_ind = i

    # If bootstrapping is desired, perform bootstrapping

    if bootstrapping is not None:
        assert type(bootstrapping) == int, "bootstrapping must be an integer"
        assert X_all is not None, "X_all must be provided if bootstrapping is desired"
        y_bootstrapped = np.zeros((bootstrapping, len(X_all)))
        for i in range(bootstrapping):
            samples = np.random.choice(np.arange(len(X_bin), dtype=int),
                                       size=len(X_bin), replace=True)
            Xdata, ydata = X_bin[samples], y_bin[samples]
            if reg_method == 'MAP':
                opt = get_w(w_inits[final_ind], Xdata, ydata, zero_prob=zero_prob,
                                method=method,
                                jac=jac,
                                reg_method=reg_method,
                                reg=(reg[0], reg[1][final_ind][0], reg[1][final_ind][1]),
                                verbose=opt_verbose,
                                slope_bound=slope_bound)
            else:
                opt = get_w(w_inits[final_ind], Xdata, ydata, zero_prob=zero_prob,
                                method=method,
                                jac=jac,
                                reg_method=reg_method,
                                reg=reg,
                                verbose=opt_verbose,
                                slope_bound=slope_bound,
                                X_orig=X_orig, y_orig=y_orig)
            
            params = opt[0]
            probs_pred = sigmoidND_nonlinear(sm.add_constant(X_all, has_constant='add'), 
                                             params)
            y_bootstrapped[i] = probs_pred
        
        y_bootstrapped_avg = np.mean(y_bootstrapped, axis=0)

        if reg_method == 'MAP':
            last_opt = get_w(w_inits[final_ind], sm.add_constant(X_all, has_constant='add'), 
                        y_bootstrapped_avg, zero_prob=zero_prob,
                        method=method, jac=jac, reg_method=reg_method, 
                        reg=(reg[0], reg[1][final_ind][0], reg[1][final_ind][1]),
                        verbose=opt_verbose, slope_bound=slope_bound)
        else:
            last_opt = get_w(w_inits[final_ind], sm.add_constant(X_all, has_constant='add'), 
                        y_bootstrapped_avg, zero_prob=zero_prob,
                        method=method, jac=jac, reg_method=reg_method, reg=reg,
                        verbose=opt_verbose, slope_bound=slope_bound,
                        X_orig=X_orig, y_orig=y_orig)

        w_inits[final_ind] = last_opt[0]

    return last_opt, w_inits

def get_w(w_init, X, y, zero_prob=0.01, method='L-BFGS-B', jac=None,
          X_orig=None, y_orig=None,
          reg_method='l2', reg=[0.01, 0.05, 0.1, 0.5, 1.0], slope_bound=20, bias_bound=None, verbose=False,
        #   options={'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxfun': 15000}):
          options={'maxiter': 20000, 'ftol': 1e-10, 'maxfun': 20000}):
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
    nll_null = negLL_hotspot(null_weights, X, y, False, reg_method, reg[0])

    # Optimize the weight vector with MLE
    opt = minimize(negLL_hotspot, x0=w_init.ravel(), bounds=bounds,
                       args=(X, y, verbose, reg_method, reg[0]), method=method,
                        jac=jac, options=options)
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)