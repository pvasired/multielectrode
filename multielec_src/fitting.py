# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from itertools import chain, combinations
import multiprocessing as mp
import collections
import copy
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import optax
import time
import matplotlib.pyplot as plt
from jaxopt import ProximalGradient, ProjectedGradient
from mpl_toolkits.mplot3d import Axes3D
import multielec_src.multielec_utils as mutils

def convertToBinaryClassifier(probs, num_trials, amplitudes, degree=1, 
                              interaction=True):
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
    
    y = []
    X = []
    num_trials = num_trials.astype(int) # convert to integer
    for j in range(len(amplitudes)):
        # Calculate the number of 1s and 0s for each probability
        num1s = int(np.around(probs[j] * num_trials[j], 0))
        num0s = num_trials[j] - num1s

        # Append all the amplitudes for this probability
        X.append(np.tile(amplitudes[j], (num_trials[j], 1)))

        # Append the 0s and 1s for this probability
        y.append(np.concatenate((np.ones(num1s), np.zeros(num0s))))
    
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
                   method (str): regularization method. 'l1', and 'l2' 
                           are supported.
                   reg (float): regularization parameter

    Returns:
    negLL (float): negative log likelihood of the data given the 
                   current parameters, possibly plus a regularization
                   term.
    """
    X, y, verbose, method, reg = args
    
    w = params.reshape(-1, X.shape[-1]).astype(float)

    # Negative log-likelihood calculation for hotspot activation
    prod = np.ones(len(X))
    for i in range(len(w)):
        prod *= (1 + np.exp(X @ w[i].T))
    prod -= 1

    prod[prod < 1e-10] = 1e-10  # prevent divide by 0 errors

    ### Deprecated calculation (gives same results but slower) ###
    # yPred2 = 1 / (1 + np.exp(-np.log(prod)))

    # Get predicted probability of spike using current parameters
    # response_mat = 1 / (1 + np.exp(-X @ w.T))
    # yPred = 1 - np.multiply.reduce(1 - response_mat, axis=1)

    # print(np.sum(np.absolute(yPred - yPred2)))
    # yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)
    # yPred[yPred == 0] = 0.000001
    
    # negative log likelihood for logistic
    # NLL = -np.sum(y * np.log(yPred) + (1 - y) * np.log(1 - yPred)) 
    ###

    # Calculate negative log likelihood
    NLL2 = np.sum(np.log(1 + prod) - y * np.log(prod))

    # Add the regularization penalty term if desired
    if method == 'l1':
        # penalty term according to l1 regularization
        penalty = reg*np.linalg.norm(w.flatten(), ord=1)
    elif method == 'l2':
        # penalty term according to l2 regularization
        penalty = reg/2*np.linalg.norm(w.flatten())**2
    elif method == 'MAP':
        regmap, mu, cov = reg
        penalty = regmap * len(X) * 0.5 * (params - mu) @ np.linalg.inv(cov) @ (params - mu)
    else:
        penalty = 0

    if verbose:
        print(NLL2, penalty)
        
    return(NLL2 + penalty)

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
                                 is supported.
                   reg (float): regularization parameter

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

    elif method == 'MAP':
        regmap, mu, cov = reg
        grad += regmap * len(X) * (np.linalg.inv(cov) @ (params - mu)).flatten()

    return grad

# Deprecated
def fsigmoid(X, w):
    """
    N-dimensional sigmoid function.
    
    Parameters:
    X (M x N np.ndarray): Input values
    w (N x 1 np.ndarray): Weight vector across input values
    
    Returns:
    sigmoid: 1 / (1 + exp(-X.w))
    """
    return 1.0 / (1.0 + np.exp(-X @ w))

def disambiguate_sigmoid(sigmoid_, spont_limit = 0.3, noise_limit = 0.0, thr_prob=0.5):
    """
    Utility for disambiguating 0/1 probability for g-sort output.
    The function converts 0s to 1s if the maximum probability 
    exceeds some spontaneous limit. For all amplitudes with
    magnitude greater than the amplitude that reached the 
    spontaneous limit, all probabilities below some noise 
    threshold are converted from 0s to 1s.
    
    Parameters:
    sigmoid_ (np.ndarray): Array of probabilities sorted according to
                           increasing magnitudes of amplitudes
    spont_limit (float): Maximum spontaneous activity threshold
    noise_limit (float): Threshold below which 0s are converted to 1s
    thr_prob (float): Value for determining where to start flipping
                      probabilities from 0 to 1

    Returns:
    sigmoid (np.ndarray): 0/1 disambiguated sigmoid with same shape
                          as sigmoid_
    
    """
    sigmoid = copy.copy(sigmoid_)
    if np.max(sigmoid) < thr_prob:
        return sigmoid
    above_limit = np.argwhere(sigmoid >= thr_prob).flatten()

    min_ind = np.amin(above_limit)
    sigmoid[min_ind:][sigmoid[min_ind:] <= noise_limit] = 1
    
    # i = np.argmin(np.abs(sigmoid[above_limit]-thr_prob))
    # upper_tail = sigmoid[min(above_limit[i] + 1, len(sigmoid) - 1):]
    # upper_tail[upper_tail<=noise_limit] = 1
    
    # sigmoid[min(above_limit[i] + 1, len(sigmoid) - 1):] = upper_tail
    return sigmoid

# Need to check modifying array in this:
def disambiguate_fitting(X_expt_, probs_, T_, w_inits,
                         verbose=False, thr=0.5, pm=0.3,
                         spont_limit=0.2):

    X_expt = copy.deepcopy(X_expt_)
    probs = copy.deepcopy(probs_)
    T = copy.deepcopy(T_)

    good_inds = np.where((probs > spont_limit) & (probs != 1))[0]
    zero_inds = np.where((probs <= spont_limit) | (probs == 1))[0]

    if len(zero_inds) > 0:
        params, _, _ = fit_surface(X_expt[good_inds], probs[good_inds], 
                                    T[good_inds], w_inits,
                                    verbose=verbose)

        probs_pred_zero = sigmoidND_nonlinear(sm.add_constant(X_expt[zero_inds], 
                                                            has_constant='add'),
                                                params)
        
        good_zero_inds = zero_inds[np.where(probs_pred_zero < thr - pm)[0]]
        good_inds_tot = np.concatenate([good_inds, good_zero_inds])

        return X_expt[good_inds_tot], probs[good_inds_tot], T[good_inds_tot]

    else:
        return X_expt, probs, T

    # # fig = plt.figure()
    # # fig.clear()
    # # ax = Axes3D(fig, auto_add_to_figure=False)
    # # fig.add_axes(ax)
    # # plt.xlabel(r'$I_1$ ($\mu$A)', fontsize=16)
    # # plt.ylabel(r'$I_2$ ($\mu$A)', fontsize=16)
    # # plt.xlim(-1.8, 1.8)
    # # plt.ylim(-1.8, 1.8)
    # # ax.set_zlim(-1.8, 1.8)
    # # ax.set_zlabel(r'$I_3$ ($\mu$A)', fontsize=16)

    # # scat = ax.scatter(X_expt[:, 0], 
    # #             X_expt[:, 1],
    # #             X_expt[:, 2], marker='o', c=probs_pred, s=20, alpha=0.8, vmin=0, vmax=1)

    # # plt.show()

    # flip_inds = np.setdiff1d(np.arange(len(probs), dtype=int), mid_inds)
    # for i in flip_inds:
    #     LL_flip = (1 - probs[i]) * np.log(probs_pred[i]) + probs[i] * np.log(1 - probs_pred[i])
    #     LL_stay = probs[i] * np.log(probs_pred[i]) + (1 - probs[i]) * np.log(1 - probs_pred[i])

    #     if LL_flip > LL_stay:
    #         probs[i] = 1 - probs[i]

    # return probs


@jax.jit
def activation_probs(x, w):
    """
    Activation probabilities using hotspot model.

    Parameters:
    w (n x d jnp.DeviceArray): Site weights
    x (c x d jnp.DeviceArray): Current levels

    Returns:
    p (c x 1 jnp.DeviceArray): Predicted probabilities
    """
    # w : site weights, n x d
    # x : current levels, c x d
    site_activations = jnp.dot(w, jnp.transpose(x)) # dimensions: n x c
    p_sites = jax.nn.sigmoid(site_activations) # dimensions : n x c
    p = 1 - jnp.prod(1 - p_sites, 0)  # dimensions: c

    return p

# @jax.jit
# def activation_probs_jac(x, w):
#     prod = jnp.prod((1 / (1 + jnp.exp(jnp.dot(w, jnp.transpose(x))))), 0)
#     sigmoid = 1 / (1 + jnp.exp(-jnp.dot(w, jnp.transpose(x))))
#     # maybe clip something here?
#     # sigmoid = jnp.clip(sigmoid, a_min=1e-5, a_max=1-1e-5)
#     # prod = jnp.clip(prod, a_min=1e-5, a_max=1-1e-5)
    
#     jac_list = []
#     for i in range(len(sigmoid)):
#         jac_list.append(x * (sigmoid[i] * prod)[:, None])

#     return jnp.hstack(jac_list)

@jax.jit
def fisher_loss_array(probs_vec, transform_mat, jac_full, trials):
    """
    Compute the Fisher loss across the entire array.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    return jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)))
    # return jnp.sum(jnp.multiply(jac_full.T, jnp.dot(jnp.linalg.inv(I_w), jac_full.T)))

@jax.jit
def fisher_loss_max(probs_vec, transform_mat, jac_full, trials):
    """
    Compute the Fisher loss across the entire array.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    sum_probs = jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)), axis=0)
    sum_cells = jnp.reshape(sum_probs, (-1, trials.shape[1])).sum(axis=-1)

    # return jnp.max(sum_cells)
    return jax.scipy.special.logsumexp(sum_cells)

@jax.jit
def fisher_loss_max_sparse(probs_vec, transform_mat, jac_full, trials):
    """
    Compute the Fisher loss across the entire array.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)

    I_w_sp = sparse.CSR.fromdense(I_w)
    jac_full_sp = sparse.CSR.fromdense(jac_full)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    sum_probs = jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)), axis=0)
    sum_cells = jnp.reshape(sum_probs, (-1, trials.shape[1])).sum(axis=-1)

    return jax.scipy.special.logsumexp(sum_cells)

@jax.jit
def fisher_loss_array_jaxopt(x, data):
    """
    Compute the Fisher loss across the entire array.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss
    """
    probs_vec, transform_mat, jac_full, T_prev = data
    trials = T_prev + x
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    t = jnp.dot(transform_mat, trials).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    return jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)))
    # return jnp.sum(jnp.multiply(jac_full.T, jnp.dot(jnp.linalg.inv(I_w), jac_full.T)))

def optimize_fisher_array(jac_full, probs_vec, transform_mat, T_prev, T, reg=None, 
                          step_size=1, n_steps=100, T_budget=5000, verbose=True):
    """
    Fisher optimization loop using optax and AdamW optimizer.

    Parameters:
    jac_full (jnp.DeviceArray): Full precomputed Jacobian matrix
    probs_vec (jnp.DeviceArray): Flattened array of all probabilities
                                 for all non-degenerate (cell, pattern)
                                 combinations
    transform_mat (jnp.DeviceArray): transformation matrix to convert trials
                                     array into correct shape matrix for 
                                     multiple cells on the same pattern
    T_prev (jnp.DeviceArray): The previously sampled trials array
    T (jnp.DeviceArray): The initialization for the to-be-optimized trials

    Returns:
    losses (np.ndarray): An array of losses per iteration of the optimization routine
    T (jnp.DeviceArray): The optimized trials matrix 
    """
    # Exponential decay of the learning rate.
    # scheduler = optax.exponential_decay(
    #     init_value=step_size, 
    #     transition_steps=1000,
    #     decay_rate=0.99,
    #     staircase=False)

    # Initialize the optimizer
    optimizer = optax.adamw(step_size)
    # optimizer = optax.lion(step_size/3)
    # optimizer = optax.sgd(learning_rate=scheduler)
    opt_state = optimizer.init(T)

    if reg is None:
        init_function = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
        reg = init_function / 20000 # 20000 worked, 100000 too large

    # Update function for computing the gradient
    @jax.jit
    def update(jac_full, probs_vec, transform_mat, T_prev, T):
        # Adding special l1-regularization term that controls the total trial budget
        fisher_lambda = lambda T, jac_full, probs_vec, transform_mat, T_prev: fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T)) + reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget)
        grads = jax.grad(fisher_lambda)(T, jac_full, probs_vec, transform_mat, T_prev)
        
        return grads
    
    losses = []
    for step in range(n_steps):
        if verbose:
            print(step)

        # Update the optimizer
        # start_grad = time.time()
        grads = update(jac_full, probs_vec, transform_mat, T_prev, T)
        # print(time.time() - start_grad)

        # start_update = time.time()
        updates, opt_state = optimizer.update(grads, opt_state, params=T)
        # print(time.time() - start_update)

        # start_apply = time.time()
        T = optax.apply_updates(T, updates)
        # print(time.time() - start_apply)
        
        # start_verbose = time.time()
        # If desired, compute the losses and store them
        if verbose:
            loss = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T))
            loss_tuple = (loss, jnp.sum(jnp.absolute(T)), loss + reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget),
                            reg * jnp.absolute(jnp.sum(jnp.absolute(T)) - T_budget))
            print(loss_tuple)
            losses += [loss_tuple]
        # print(time.time() - start_verbose)

    return np.array(losses), T

def fisher_sampling_1elec(probs_empirical, T_prev, amps, w_inits_array=None, t_final=None, 
                          budget=10000, reg=None, T_step_size=0.05, T_n_steps=5000, ms=[1, 2],
                          verbose=True, pass_inds=None, R2_cutoff=0, return_probs=False,
                          disambiguate=True, empty_trials=1, min_prob=0.2, min_inds=50,
                          priors_array=None, regmap=None, trial_cap=25, entropy_buffer=0.5,
                          entropy_samples=5, exploit_factor=0.75, data_1elec_array=None,
                          min_clean_inds=20):

    """
    Parameters:
    probs_empirical: cells x patterns x amplitudes numpy.ndarray of probabilities from g-sort
    T_prev: patterns x amplitudes numpy.ndarray of trials that have already been done
    amps: patterns x amplitudes x stimElecs numpy.ndarray of current amplitudes applied
    w_inits_array: cells x patterns numpy.ndarray(dtype=object) of lists containing initial guesses for fits
    t_final: numpy.ndarray of last optimal trial allocation

    Returns:
    T_new: patterns x amplitudes numpy.ndarray of new trials to perform
    w_inits_array: cells x patterns numpy.ndarray(dtype=object) of lists containing new initial guesses for fits
    t_final: numpy.ndarray of new optimal trial allocation
    """

    print('Setting up data...')

    if w_inits_array is None:
        w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
        for i in range(len(w_inits_array)):
            for j in range(len(w_inits_array[i])):
                w_inits = []

                for m in ms:
                    w_init = np.array(np.random.normal(size=(m, amps[j].shape[1]+1)))
                    w_inits.append(w_init)

                w_inits_array[i][j] = w_inits

    print('Fitting dataset...')

    input_list = generate_input_list(probs_empirical, amps, T_prev, w_inits_array, min_prob,
                                        priors_array=priors_array, regmap=regmap,
                                        pass_inds=pass_inds, disambiguate=disambiguate,
                                        min_inds=min_inds, data_1elec_array=data_1elec_array,
                                        min_clean_inds=min_clean_inds)

    pool = mp.Pool(processes=24)
    results = pool.starmap_async(fit_surface, input_list)
    mp_output = results.get()
    pool.close()

    params_curr = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    R2s = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]))
    probs_curr = np.zeros(probs_empirical.shape)

    cnt = 0
    for i in range(len(probs_empirical)):
        for j in range(len(probs_empirical[i])):
            params_curr[i][j] = mp_output[cnt][0]
            w_inits_array[i][j] = mp_output[cnt][1]
            R2s[i][j] = mp_output[cnt][2]
            
            probs_curr[i][j] = sigmoidND_nonlinear(
                                    sm.add_constant(amps[j], has_constant='add'), 
                                    params_curr[i][j])

            cnt += 1

    print('Calculating Jacobian...')

    jac_dict = collections.defaultdict(dict)
    transform_mat = []
    probs_vec = []
    num_params = 0
    entropy_inds = []

    for i in range(len(params_curr)):
        for j in range(len(params_curr[i])):
            if ~np.all(params_curr[i][j][:, 0] == -np.inf) and R2s[i][j] >= R2_cutoff:
                X = jnp.array(sm.add_constant(amps[j], has_constant='add'))
                # jac_dict[i][j] = activation_probs_jac(X, jnp.array(params_curr[i][j]))
                jac_dict[i][j] = jax.jacfwd(activation_probs, argnums=1)(X, jnp.array(params_curr[i][j])).reshape(
                                                (len(X), params_curr[i][j].shape[0]*params_curr[i][j].shape[1]))  # c x l
                num_params += jac_dict[i][j].shape[1]

                probs_pred = sigmoidND_nonlinear(sm.add_constant(amps[j], 
                                                    has_constant='add'), 
                                                    params_curr[i][j])
                entropy_inds_j = np.where((probs_pred >= 0.5 - entropy_buffer) & (probs_pred <= 0.5 + entropy_buffer))[0]
                for ind in entropy_inds_j:
                    entropy_inds.append((j, ind))

                transform = jnp.zeros(len(T_prev))
                transform = transform.at[j].set(1)
                transform_mat.append(transform)     # append a e-vector (512)

                probs_vec.append(probs_curr[i][j])  # append a c-vector (80)

    if len(probs_vec) == 0:
        raise ValueError("No valid probabilities found.")
        # if return_probs:
        #     return np.ones_like(T_prev, dtype=int) * empty_trials, w_inits_array, np.ones_like(T_prev, dtype=int) * empty_trials, probs_curr, params_curr
        
        # else:
        #     return np.ones_like(T_prev, dtype=int) * empty_trials, w_inits_array, np.ones_like(T_prev, dtype=int) * empty_trials

    entropy_inds = np.array(entropy_inds)
    transform_mat = jnp.array(transform_mat, dtype='float32')
    probs_vec = jnp.array(jnp.hstack(probs_vec), dtype='float32')

    jac_full = jnp.zeros((len(probs_vec), num_params))
    counter_axis0 = 0
    counter_axis1 = 0
    for i in jac_dict.keys():
        for j in jac_dict[i].keys():
            next_jac = jac_dict[i][j]

            jac_full = jac_full.at[counter_axis0:counter_axis0+next_jac.shape[0], counter_axis1:counter_axis1+next_jac.shape[1]].set(next_jac)

            counter_axis0 += next_jac.shape[0]
            counter_axis1 += next_jac.shape[1]

    jac_full = jnp.array(jac_full, dtype='float32')
    print('Optimizing trials...')

    if t_final is None:
        # T_new_init = jnp.ones_like(jnp.array(T_prev), dtype='float32')
        random_init = np.random.choice(len(T_prev.flatten()), size=int(budget*exploit_factor))
        T_new_init = jnp.array(np.bincount(random_init, minlength=len(T_prev.flatten())).astype(int).reshape(T_prev.shape), dtype='float32')

    else:
        T_new_init = jnp.array(jnp.absolute(jnp.array(t_final)), dtype='float32')

    # print(fisher_loss_array(probs_vec, transform_mat, jac_full, jnp.array(T_prev, dtype='float32')))
    losses, t_final = optimize_fisher_array(jac_full, probs_vec, transform_mat, jnp.array(T_prev, dtype='float32'), T_new_init, 
                                                    step_size=T_step_size, n_steps=T_n_steps, reg=reg, T_budget=budget*exploit_factor,
                                                    verbose=verbose)

    if verbose:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(losses[:, 0])
        axs[0].set_ylabel('Fisher Loss (A-optimality)')
        axs[1].plot(losses[:, 1])
        axs[1].set_ylabel('Total Trials')
        axs[2].plot(losses[:, 2])
        axs[2].set_ylabel('Regularized Loss, reg=' + str(reg))

        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        plt.savefig(f'plots_CL.png', dpi=300)
        plt.show(block=False)

    T_new = jnp.round(jnp.absolute(t_final), 0)

    # if jnp.sum(T_new) < budget:
    #     random_extra = np.random.choice(len(T_new.flatten()), size=int(budget - jnp.sum(T_new)),
    #                                     p=np.array(jnp.absolute(t_final.flatten()))/np.sum(np.array(jnp.absolute(t_final.flatten()))))
    #     T_new_extra = jnp.array(np.bincount(random_extra, minlength=len(T_new.flatten())).astype(int).reshape(T_new.shape), dtype='float32')
    #     T_new = T_new + T_new_extra

    T_new = np.array(T_new)
    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if np.sum(T_new) < budget:
        random_entropy = np.random.choice(len(entropy_inds), 
                                          size=int((budget - np.sum(T_new))/entropy_samples))
        for ind in random_entropy:
            T_new[entropy_inds[ind][0]][entropy_inds[ind][1]] += entropy_samples

    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if return_probs:
        return T_new.astype(int), w_inits_array, np.array(t_final), probs_curr, params_curr
    
    else:
        return T_new.astype(int), w_inits_array, np.array(t_final)

# Deprecated
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
                        priors_array=None, regmap=None, data_1elec_array=None,
                        pass_inds=None, disambiguate=True, min_inds=50,
                        min_clean_inds=20, spont_limit=0.2):
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
    input_list (list): formatter list ready for multiprocessing
    """
    all_probs = copy.deepcopy(all_probs_)
    amps = copy.deepcopy(amps_)
    trials = copy.deepcopy(trials_)

    input_list = []
    for i in range(len(all_probs)):
        for j in range(len(all_probs[i])):
            if pass_inds is not None:
                if pass_inds[i][j] is not None:
                    probs = all_probs[i][j][pass_inds[i][j]]
                    T = trials[j][pass_inds[i][j]]
                    X = amps[j][pass_inds[i][j]]

                else:
                    probs = np.zeros(len(all_probs[i][j]))
                    T = trials[j]
                    X = amps[j]

            else:
                probs = all_probs[i][j]
                T = trials[j]
                X = amps[j]

            if not(disambiguate):
                good_T_inds = np.where(T > 0)[0]

                probs = probs[good_T_inds]
                T = T[good_T_inds]
                X = X[good_T_inds]

                good_inds = np.where((probs > spont_limit) & (probs != 1))[0]

                if len(good_inds) >= min_inds:
                    clean_inds = mutils.triplet_cleaning(X, probs, T, return_inds=True)
                    above_spont = np.where(probs[clean_inds] >= spont_limit)[0]
                    if len(above_spont) < min_clean_inds:
                        probs = np.array([])
                        X = np.array([])
                        T = np.array([])

                    else:
                        dirty_inds = np.setdiff1d(np.arange(len(X), dtype=int),
                                                clean_inds)
                        probs[dirty_inds] = 0

                        X, probs, T = disambiguate_fitting(X, probs, T, w_inits_array[i][j])

                else:
                    probs = np.array([])
                    X = np.array([])
                    T = np.array([])
            
            else:
                good_T_inds = np.where(T > 0)[0]

                probs = probs[good_T_inds]
                T = T[good_T_inds]
                X = X[good_T_inds]
                
            if len(probs[probs > min_prob]) == 0:
                probs = np.array([])
                X = np.array([])
                T = np.array([])

            if data_1elec_array is not None and data_1elec_array[i][j] != 0:
                if len(X) > 0:
                    X = np.vstack((X, data_1elec_array[i][j][0]))
                    probs = np.hstack((probs, data_1elec_array[i][j][1]))
                    T = np.hstack((T, data_1elec_array[i][j][2]))
                else:
                    X, probs, T = data_1elec_array[i][j]

            if priors_array is None or priors_array[i][j] == 0:
                input_list += [(X, probs, T, w_inits_array[i][j])]
            
            else:
                input_list += [(X, probs, T, w_inits_array[i][j],
                                'MAP', (regmap, priors_array[i][j]))]

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

def enforce_3D_monotonicity(index, Xdata, ydata, k=2, 
                            percentile=0.9, num_points=20,
                            dist_thr=0.3):
    point = Xdata[index]
    if np.linalg.norm(point) == 0:
        return True
        
    direction = point / np.linalg.norm(point)

    scaling = np.linspace(0, np.linalg.norm(point), num_points)
    closest_line = []
    for j in range(len(scaling)):
        curr = scaling[j] * direction
        dists = cdist(Xdata, curr[:, None].T).flatten()
        closest_inds = np.setdiff1d(np.where(dists <= dist_thr)[0], index)
        # closest_inds = np.setdiff1d(np.argsort(dists)[:k], index)

        closest_line.append(closest_inds)

    line_inds = np.unique(np.concatenate(closest_line))

    if len(line_inds) > 0:
        if ydata[index] >= percentile * np.amax(ydata[line_inds]):
            return True
        
        else:
            return False
    
    else:
        return False

def fit_surface(X_expt, probs, T, w_inits, reg_method='none', reg=0,
                        R2_thresh=0.1, zero_prob=0.01, verbose=False,
                        method='L-BFGS-B', jac=negLL_hotspot_jac,
                        opt_verbose=False):
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
    X_orig = copy.copy(X_expt)

    # If the probability never gets large enough, return the degenerate parameters
    # The degenerate parameters are a bias term of -np.inf and all slopes set to 0
    # These parameters cause probs_pred to be an array of all 0s
    if len(probs) == 0:
        deg_opt = np.zeros_like(w_inits[-1])
        deg_opt[:, 0] = np.ones(len(deg_opt)) * -np.inf

        return deg_opt, w_inits, -1
    
    # If a large enough probability was detected, begin fitting

    # Convert the data to binary classification data
    X_bin, y_bin = convertToBinaryClassifier(probs, T, X_expt)

    # Compute the negative log likelihood of the null model which only
    # includes an intercept
    ybar = np.mean(y_bin)
    beta_null = np.log(ybar / (1 - ybar))
    null_weights = np.concatenate((np.array([beta_null]), 
                                   np.zeros(X_expt.shape[-1])))
    nll_null = negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 
                             0)
    if verbose:
        print(nll_null)

    # Now begin the McFadden pseudo-R2 early stopping loop
    if reg_method == 'MAP':
        last_opt = get_w(w_inits[0], X_bin, y_bin, nll_null, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, 
                        reg=(reg[0], reg[1][0][0], reg[1][0][1]),
                        verbose=opt_verbose)
    else:
        last_opt = get_w(w_inits[0], X_bin, y_bin, nll_null, zero_prob=zero_prob, 
                        method=method, jac=jac, reg_method=reg_method, reg=reg, verbose=opt_verbose)
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
            new_opt = get_w(w_inits[i], X_bin, y_bin, nll_null, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=(reg[0], reg[1][i][0], reg[1][i][1]),
                            verbose=opt_verbose)
        else:
            new_opt = get_w(w_inits[i], X_bin, y_bin, nll_null, zero_prob=zero_prob,
                            method=method,
                            jac=jac,
                            reg_method=reg_method,
                            reg=reg,
                            verbose=opt_verbose)
        w_inits[i] = new_opt[0]
        new_R2 = new_opt[2]
        BIC = len(w_inits[i].flatten()) * np.log(len(X_bin)) + 2 * new_opt[1]
        HQC = 2 * len(w_inits[i].flatten()) * np.log(np.log(len(X_bin))) + 2 * new_opt[1]

        if verbose:
            print(new_opt, new_R2, BIC, HQC)

        # If the pseudo-R2 improvement was too small, break and stop adding sites
        # if new_R2 - last_R2 <= R2_thresh:
        #     break

        if last_R2 > 0 and (new_R2 - last_R2) / last_R2 <= R2_thresh:
            break

        last_opt = new_opt
        last_R2 = new_R2

    return last_opt[0], w_inits, last_R2

def get_w(w_init, X, y, nll_null, zero_prob=0.01, method='L-BFGS-B', jac=None,
          reg_method='none', reg=0, slope_bound=10, bias_bound=None, verbose=False):
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

    # Optimize the weight vector with MLE
    opt = minimize(negLL_hotspot, x0=w_init.ravel(), bounds=bounds,
                       args=(X, y, verbose, reg_method, reg), method=method,
                        jac=jac)
    
    return opt.x.reshape(-1, X.shape[-1]), opt.fun, (1 - opt.fun / nll_null)

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
