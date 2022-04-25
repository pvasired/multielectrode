import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
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
