import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def convertToBinaryClassifier(probs, num_trials, amplitudes, degree=1):
    y = []
    X = []
    for j in range(len(amplitudes)):
        num1s = int(np.around(probs[j] * num_trials[j], 0))
        num0s = num_trials[j] - num1s

        X.append(np.tile(amplitudes[j], (num_trials[j], 1)))
        y.append(np.concatenate((np.ones(num1s), np.zeros(num0s))))
        
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(np.concatenate(X))
    y = np.concatenate(y)
    
    return X, y

def negLL(params, *args):
    X, y, verbose, method = args
    
    w = params
    
    # Get predicted probability of spike using current parameters
    yPred = 1 / (1 + np.exp(-X @ w))
    yPred[yPred == 1] = 0.999999     # some errors when yPred is exactly 1 due to taking log(1 - 1)

    # Calculate negative log likelihood
    NLL = -np.sum(y * np.log2(yPred) + (1 - y) * np.log2(1 - yPred))     # negative log likelihood for logistic

    if method == 'MAP':
         penalty = 0.5 * (w - mu) @ np.linalg.inv(cov) @ (w - mu)     # penalty term according to MAP regularization
    else:
         penalty = 0

    if verbose:
        print(NLL, penalty)
    return(NLL + penalty)

def fsigmoid(X, w):
    return 1.0 / (1.0 + np.exp(-X @ w))
