import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import src.fitting as fitting
import src.multielec_utils as mutils
import statsmodels.api as sm
import jax
import jax.numpy as jnp
import optax

def activation_probs(x, w):
    # w : site weights, n x d
    # x : current levels, c x d
    site_activations = jnp.dot(w, jnp.transpose(x)) # dimensions: n x c
    p_sites = jax.nn.sigmoid(site_activations) # dimensions : n x c
    p = 1 - jnp.prod(1 - p_sites, 0)  # dimensions: c

    return p

def neg_log_likelihood(w, X, y, l2_reg=0):
    # x : current levels, c x d
    # w : site weights, n x d
    # y : empirical probability for each current level, c
    # trials: number of trials at each current level, c
    # l2_reg: l2 regularization penalty
    
    # Get predicted probability of spike using current parameters
    yPred = activation_probs(X, w)
    yPred = jnp.clip(yPred, a_min=1e-5, a_max=1-1e-5)

    NLL = -jnp.sum(y * jnp.log(yPred) + (1 - y) * jnp.log(1 - yPred))   # negative log likelihood for logistic

    penalty = l2_reg/2 * jnp.linalg.norm(w)**2

    return NLL + penalty

def optimize_w(w, X, y, l2_reg=0, zero_prob=0.01, step_size=0.0001, n_steps=100, wtol=5e-5, step_cnt_decrement=5):

    m = len(w)
    z = 1 - (1 - zero_prob)**(1/m)

    optimizer = optax.adamw(step_size)
    opt_state = optimizer.init(w)

    @jax.jit
    def update(w, X, y, l2_reg):
        grads = jax.grad(neg_log_likelihood)(w, X, y, l2_reg=l2_reg)
        return grads

    losses = []
    prev_w = w
    for step in range(n_steps):
        grads = update(w, X, y, l2_reg)
        updates, opt_state = optimizer.update(grads, opt_state, params=w)
        w = optax.apply_updates(w, updates)

        losses += [neg_log_likelihood(w, X, y, l2_reg=l2_reg)]
        w = w.at[:, 0].set(jnp.minimum(w[:, 0], np.log(z/(1-z))))

        if jnp.linalg.norm(w - prev_w) / len(w.ravel()) <= wtol:
            break
        prev_w = w
        
    return losses, w

def fit_surface_McF(X, y, w_inits, R2_thresh=0.02, l2_reg=0.01, w_step_size=0.001, n_steps=3500, plot=False, random_state=None):
    
    ybar = jnp.mean(y)
    beta_null = jnp.log(ybar / (1 - ybar))
    null_weights = jnp.concatenate((jnp.array([beta_null]), jnp.zeros(X.shape[-1]-1)))
    nll_null = neg_log_likelihood(null_weights, X, y, l2_reg=l2_reg)
    
    losses, w_final = optimize_w(w_inits[0], X, y, l2_reg=l2_reg, step_size=w_step_size, n_steps=n_steps)

    last_opt = w_final
    last_R2 = 1 - losses[-1] / nll_null
    w_inits[0] = w_final

    for i in range(1, len(w_inits)):
        losses, w_final = optimize_w(w_inits[i], X, y, l2_reg=l2_reg, step_size=w_step_size, n_steps=n_steps)
        w_inits[i] = w_final

        new_opt = w_final
        new_R2 = 1 - losses[-1] / nll_null
        if new_R2 - last_R2 <= R2_thresh:
            break

        last_opt = new_opt
        last_R2 = new_R2

    w_final = last_opt

    return w_final, w_inits

def fisher_info(x, w, t):
    # x : current levels, c x d
    # w : site weights, n x d
    # y : empirical probability for each current level, c
    # t: number of trials for each current level, c
    
    p_model = jnp.clip(activation_probs(x, w), a_min=1e-5, a_max=1-1e-5) # c
    I_p = jnp.diag(t / (p_model * (1 - p_model)))   # c x c
    J = jax.jacfwd(activation_probs, argnums=1)(x, w).reshape((len(x), w.shape[0]*w.shape[1]))
    I_w = jnp.dot(jnp.dot(J.T, I_p), J) / len(x)
    
    loss = jnp.trace(J @ (jnp.linalg.inv(I_w) @ J.T))
    return loss

def optimize_fisher(x, w, t_prev, t, reg=0, step_size=0.001, n_steps=100, reltol=-np.inf, T_budget=5000, step_cnt_decrement=5):

    optimizer = optax.adamw(step_size)
    opt_state = optimizer.init(t)

    @jax.jit
    def update(x, w, t_prev, t):
        fisher_lambda = lambda t, x, w, t_prev: fisher_info(x, w, t_prev + jnp.absolute(t))  + reg * jnp.absolute(jnp.sum(jnp.absolute(t)) - T_budget)
        grads = jax.grad(fisher_lambda)(t, x, w, t_prev)

        return grads
    
    losses = []
    for step in range(n_steps):
        grads = update(x, w, t_prev, t)
        updates, opt_state = optimizer.update(grads, opt_state, params=t)
        t = optax.apply_updates(t, updates)
    
        losses += [[fisher_info(x, w, t_prev + jnp.absolute(t)), 
                    jnp.sum(jnp.absolute(t)),
                    fisher_info(x, w, t_prev + jnp.absolute(t)) + reg * jnp.absolute(jnp.sum(jnp.absolute(t)) - T_budget)]]

    return np.array(losses), t

ms = [1, 2]
l2_reg = 0
w_step_size = 0.1
n_steps = 1000
R2_thresh = 0.002

total_budget = 200
num_iters = 5
budget = int(total_budget / num_iters)
reg = 1
T_step_size = 0.1
T_n_steps = 5000

# Need amplitudes (X), trials (T_prev), probabilities (p_empirical), w_inits, T_new_init

###
# FIRST STEP
w_inits = []
for m in ms:
    w_init = jnp.array(np.random.normal(size=(m, X.shape[1]+1)))
    w_inits.append(w_init)

if cnt == 0:
    T_new_init = jnp.zeros(len(T_prev)) + 1
else:
    T_new_init = t_final
###

# Single step
good_inds = jnp.where((p_empirical > 0) & (p_empirical < 1))[0]
X_bin, y_bin = fitting.convertToBinaryClassifier(np.array(p_empirical[good_inds]), 
                                                    np.array(T_prev[good_inds], dtype=int),
                                                    np.array(X[good_inds]))
X_bin = jnp.array(X_bin)
y_bin = jnp.array(y_bin)

w_final, w_inits =  fit_surface_McF(X_bin, y_bin, w_inits, R2_thresh=R2_thresh, 
                                    l2_reg=l2_reg, w_step_size=w_step_size, n_steps=n_steps, plot=False)

losses, t_final = optimize_fisher(jnp.array(sm.add_constant(X, has_constant='add')), w_final, T_prev, T_new_init, 
                                    reg=reg, step_size=T_step_size, n_steps=T_n_steps, T_budget=budget)
T_new = jnp.round(jnp.absolute(t_final), 0)
if jnp.sum(T_new) < budget:
    random_extra = np.random.choice(len(X), size=int(budget - jnp.sum(T_new)), 
                                    p=np.array(jnp.absolute(t_final))/np.sum(np.array(jnp.absolute(t_final))))
    T_new_extra = jnp.array(np.bincount(random_extra, minlength=len(X))).astype(int)
    T_new = T_new + T_new_extra

# Output: T_new, update w_inits and T_new_init