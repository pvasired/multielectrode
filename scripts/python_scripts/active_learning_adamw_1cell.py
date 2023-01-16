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
from scipy.io import savemat, loadmat
import optax
from copy import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, help='Estim run in format dataXXX.')
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
parser.add_argument('-p', '--pattern', type=int, help="Estim pattern.")
parser.add_argument('-c', '--cell', type=int, help="Cell of interest.")
parser.add_argument('-b', '--budget', type=int, help="Total trial budget.")
parser.add_argument('-s', '--steps', type=int, help="Number of active steps.")
parser.add_argument('-r', '--restarts', type=int, help="Number of restarts.")
args = parser.parse_args()

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

def fit_surface_McF(X, y, w_inits, R2_thresh=0.02, l2_reg=0.01, w_step_size=0.001, n_steps=3500, plot=False):
    
    ybar = jnp.mean(y)
    beta_null = jnp.log(ybar / (1 - ybar))
    null_weights = jnp.concatenate((jnp.array([beta_null]), jnp.zeros(X.shape[-1]-1)))
    nll_null = neg_log_likelihood(null_weights, X, y, l2_reg=l2_reg)
    print(nll_null)
    
    losses, w_final = optimize_w(w_inits[0], X, y, l2_reg=l2_reg, step_size=w_step_size, n_steps=n_steps)
    last_opt = w_final
    last_R2 = 1 - losses[-1] / nll_null
    w_inits[0] = w_final
    print(last_R2, losses[-1])

    for i in range(1, len(w_inits)):
        losses, w_final = optimize_w(w_inits[i], X, y, l2_reg=l2_reg, step_size=w_step_size, n_steps=n_steps)
        w_inits[i] = w_final

        new_opt = w_final
        new_R2 = 1 - losses[-1] / nll_null
        print(new_R2, losses[-1])
        if new_R2 - last_R2 <= R2_thresh:
            break

        last_opt = new_opt
        last_R2 = new_R2

    w_final = last_opt

    return w_final, w_inits

def sample_spikes(p_true, t):
    p_true, t = np.array(p_true), np.array(t).astype(int)
    
    p_empirical = []
    for i in range(len(p_true)):
        if t[i] == 0:
            p_empirical += [0.5]
        
        else:
            p_empirical += [np.mean(np.random.choice(np.array([0, 1]), 
                                                 p=np.array([1-p_true[i], p_true[i]]), 
                                                 size=t[i]))]
        
    return p_empirical

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

def get_performance_AL(X, w_meas, p_true):
    probs_pred = activation_probs(X, w_meas)
    RMSE = jnp.sqrt(jnp.sum((probs_pred - p_true)**2) / len(X))

    return RMSE

# Path definitions
ANALYSIS_BASE = "/Volumes/Analysis"
MATFILE_BASE = "/Volumes/Scratch/Users/praful/triplet_gsort_matfiles_20220420"
gsort_path = None

dataset = args.dataset
estim = args.estim
wnoise = args.wnoise
electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
vis_datapath = os.path.join(ANALYSIS_BASE, dataset, wnoise)
p = args.pattern
cell = args.cell

ms = [2, 3, 4, 5]
l2_reg = 0
w_step_size = 0.1
n_steps = 1000
R2_thresh = 0.002

X_expt_orig, probs_orig, T_orig = mutils.loadNewLVData(electrical_path, gsort_path, dataset, estim, wnoise, p, cell,
                                        load_from_mat=True, 
                                        MATFILE_BASE=MATFILE_BASE)
X_expt, probs, T = mutils.triplet_cleaning(X_expt_orig, probs_orig, T_orig, electrical_path, p)
X_bin, y_bin = fitting.convertToBinaryClassifier(probs, T, X_expt)
X_bin = jnp.array(X_bin)
y_bin = jnp.array(y_bin)

w_inits = []
for m in ms:
    w_init = jnp.array(np.random.normal(size=(m, X_bin.shape[1])))
    w_inits.append(w_init)
    
w_true, _ =  fit_surface_McF(X_bin, y_bin, w_inits, R2_thresh=R2_thresh, l2_reg=l2_reg, w_step_size=w_step_size, n_steps=n_steps)

print(w_true)
X_expt_orig = mutils.get_stim_amps_newlv(electrical_path, p)
p_true = activation_probs(jnp.array(sm.add_constant(X_expt_orig, has_constant='add')), w_true) # prob with each current level
X = jnp.array(X_expt_orig)

total_budget = args.budget
num_iters = args.steps
budget = int(total_budget / num_iters)
reg = 20
num_restarts = args.restarts
T_step_size = 0.01
T_n_steps = 5000

init_size = 200
init_trials = 5

performance_stack = []
performance_stack_random = []
num_samples_stack = []
t_final_stack = []
t_final_init_stack = []
m_stack = []
m_stack_random = []

for restart in range(num_restarts):
    print('Restart', restart + 1)
    # Initialize amplitudes
    init_inds = np.random.choice(len(X), replace=False, size=init_size)

    # Initialize trials
    T_prev = jnp.zeros(len(X_expt_orig))
    T_prev = T_prev.at[init_inds].set(init_trials)
    T_prev_random = jnp.copy(T_prev)

    p_empirical = jnp.array(sample_spikes(p_true, T_prev))
    p_empirical_random = jnp.copy(p_empirical)

    performances = []
    performances_random = []
    num_samples = []
    t_finals = []
    ms_run = []
    ms_run_random = []

    w_inits = []
    for m in ms:
        w_init = jnp.array(np.random.normal(size=(m, X.shape[1]+1)))
        w_inits.append(w_init)

    w_inits_random = copy(w_inits)
    cnt = 0

    while True:
        num_samples.append(np.sum(np.absolute(np.array(T_prev)).astype(int)))

        sampled_inds = np.where(np.absolute(np.array(T_prev)).astype(int) > 0)[0]
        sampled_inds_random = np.where(np.absolute(np.array(T_prev_random)).astype(int) > 0)[0]

        X_bin, y_bin = fitting.convertToBinaryClassifier(np.array(p_empirical[sampled_inds]), 
                                                         np.array(T_prev[sampled_inds], dtype=int),
                                                         np.array(X[sampled_inds]))
        X_bin = jnp.array(X_bin)
        y_bin = jnp.array(y_bin)

        w_final, w_inits =  fit_surface_McF(X_bin, y_bin, w_inits, R2_thresh=R2_thresh, 
                                            l2_reg=l2_reg, w_step_size=w_step_size, n_steps=n_steps, plot=False)
        print(w_final)

        ms_run.append(len(w_final))
        if cnt == 0:
            T_new_init = jnp.zeros(len(T_prev)) + 1
            w_final_random = jnp.copy(w_final)
        else:
            T_new_init = t_final
            X_bin, y_bin = fitting.convertToBinaryClassifier(np.array(p_empirical_random[sampled_inds_random]), 
                                                         np.array(T_prev_random[sampled_inds_random], dtype=int),
                                                         np.array(X[sampled_inds_random]))
            X_bin = jnp.array(X_bin)
            y_bin = jnp.array(y_bin)

            w_final_random, w_inits_random =  fit_surface_McF(X_bin, y_bin, w_inits_random, R2_thresh=R2_thresh, 
                                                l2_reg=l2_reg, w_step_size=w_step_size, n_steps=n_steps, plot=False)

        print(w_final_random)
        ms_run_random.append(len(w_final_random))

        performance = get_performance_AL(jnp.array(sm.add_constant(X, has_constant='add')), w_final, p_true)
        performances.append(performance)
        
        performance_random = get_performance_AL(jnp.array(sm.add_constant(X, has_constant='add')), w_final_random, p_true)
        performances_random.append(performance_random)

        print(performance, performance_random)

        if cnt >= num_iters:
            break
        
        losses, t_final = optimize_fisher(jnp.array(sm.add_constant(X, has_constant='add')), w_final, T_prev, T_new_init, reg=reg, step_size=T_step_size, n_steps=T_n_steps, T_budget=budget)

        if cnt == 0:
            t_final_init_stack.append(jnp.absolute(t_final))
        else:
            t_finals.append(jnp.absolute(t_final))
        T_new = jnp.round(jnp.absolute(t_final), 0)
        if jnp.sum(T_new) < budget:
            random_extra = np.random.choice(len(X), size=int(budget - jnp.sum(T_new)), 
                                            p=np.array(jnp.absolute(t_final))/np.sum(np.array(jnp.absolute(t_final))))
            T_new_extra = jnp.array(np.bincount(random_extra, minlength=len(X))).astype(int)
            T_new = T_new + T_new_extra

        p_new = jnp.array(sample_spikes(p_true, T_new))

        p_tmp = (p_new * T_new + p_empirical * T_prev) / (T_prev + T_new)
        T_tmp = T_prev + T_new
        p_tmp = p_tmp.at[jnp.isnan(p_tmp)].set(0.5)

        p_empirical = p_tmp
        T_prev = T_tmp

        random_draws = np.random.choice(len(X), size=int(jnp.sum(T_new)))
        T_new_random = jnp.array(np.bincount(random_draws, minlength=len(X))).astype(int)
        p_new_random = jnp.array(sample_spikes(p_true, T_new_random))
        
        p_tmp_random = (p_new_random * T_new_random + p_empirical_random * T_prev_random) / (T_prev_random + T_new_random)
        T_tmp_random = T_prev_random + T_new_random
        p_tmp_random = p_tmp_random.at[jnp.isnan(p_tmp_random)].set(0.5)

        p_empirical_random = p_tmp_random
        T_prev_random = T_tmp_random

        cnt += 1
    
    performance_stack.append(performances)
    performance_stack_random.append(performances_random)
    num_samples_stack.append(num_samples)
    t_final_stack.append(np.array(t_finals))
    m_stack.append(ms_run)
    m_stack_random.append(ms_run_random)

savemat('active_learning-{}-n{}-p{}-r{}.mat'.format(dataset, str(cell), str(p), str(args.restarts)),
        {'dataset': dataset,
            'cell': cell,
            'pattern': p,
            'estim': electrical_path,
            'wnoise': vis_datapath,
            'budget': args.budget,
            'active_steps': args.steps,
            'restarts': args.restarts,
            'w_true': np.array(w_true),
            'X': np.array(X),
            'performance_stack': np.array(performance_stack),
            'performance_stack_random': np.array(performance_stack_random),
            'num_samples_stack': np.array(num_samples_stack),
            't_final_stack': np.array(t_final_stack),
            't_final_init_stack': np.array(t_final_init_stack),
            'm_stack': np.array(m_stack),
            'm_stack_random': np.array(m_stack_random)})

import pdb; pdb.set_trace()