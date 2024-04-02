# Utilities for fitting electrical stimulation spike sorting data

import numpy as np
import statsmodels.api as sm
import multiprocessing as mp
import collections
import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt
import multielec_src.fitting as fitting

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

@jax.jit
def activation_probs_erf(x, w):
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
    p_sites = 0.5 + 0.5*jax.scipy.special.erf(site_activations) # dimensions : n x c
    p = 1 - jnp.prod(1 - p_sites, 0)  # dimensions: c

    return p

@jax.jit
def fisher_loss_max(probs_vec, transform_mat, jac_full, trials, bundle_mask):
    """
    Compute the Fisher loss across the entire array, taking logsumexp()
    to minimize the worst case.

    Parameters:
    probs_vec (jnp.DeviceArray): The flattened array of probabilities across all meaningful
                            (cell, pattern) combinations
    transform_mat (jnp.DeviceArray): The transformation matrix to convert the trials array
                                to the transformed trials array for multiple cells on
                                the same pattern
    jac_full (jnp.DeviceArray): The full precomputed Jacobian matrix
    trials (jnp.DeviceArray): The input trials vector to be optimized

    Returns:
    loss (float): The whole array Fisher information loss, taking logsumexp() across
                  cells to minimize the worst case.
    """
    p_model = jnp.clip(probs_vec, a_min=1e-5, a_max=1-1e-5) # need to clip these to prevent
                                                            # overflow errors
    trials_masked = jnp.where(bundle_mask, trials, 0)
    t = jnp.dot(transform_mat, trials_masked).flatten()
    I_p = t / (p_model * (1 - p_model))

    # Avoiding creating the large diagonal matrix and storing in memory
    I_w = jnp.dot((jac_full.T * I_p), jac_full) / len(p_model)
    
    # Avoiding multiplying the matrices out and calculating the trace explicitly
    sum_probs = jnp.sum(jnp.multiply(jac_full.T, jnp.linalg.solve(I_w, jac_full.T)), axis=0)
    sum_cells = jnp.reshape(sum_probs, (-1, trials.shape[1])).sum(axis=-1)

    return jax.scipy.special.logsumexp(sum_cells)

def optimize_fisher_array(jac_full, probs_vec, transform_mat, T_prev, T, bundle_mask,
                          reg=None, step_size=0.01, n_steps=3000, T_budget=10000, verbose=True):
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
        init_function = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T), bundle_mask)
        reg = init_function / 1000000 # 10000 worked, 100000 too large

    # Update function for computing the gradient
    @jax.jit
    def update(jac_full, probs_vec, transform_mat, T_prev, T, bundle_mask):
        # Adding special l1-regularization term that controls the total trial budget
        fisher_lambda = lambda T, jac_full, probs_vec, transform_mat, T_prev, bundle_mask: fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T), bundle_mask) + reg * (jnp.sum(jnp.absolute(T)) - T_budget)**2

        grads = jax.grad(fisher_lambda)(T, jac_full, probs_vec, transform_mat, T_prev, bundle_mask)
        
        return grads
    
    losses = []
    for step in range(n_steps):
        if verbose:
            print(step)

        # Update the optimizer
        # start_grad = time.time()
        grads = update(jac_full, probs_vec, transform_mat, T_prev, T, bundle_mask)
        # print(time.time() - start_grad)

        # start_update = time.time()
        updates, opt_state = optimizer.update(grads, opt_state, params=T)
        # print(time.time() - start_update)

        # start_apply = time.time()
        T = optax.apply_updates(T, updates)
        # print(time.time() - start_apply)

        # Mask the trials at bundle_mask to be zero
        T = jnp.where(bundle_mask, T, 0)
        
        # start_verbose = time.time()
        # If desired, compute the losses and store them
        if verbose:
            loss = fisher_loss_max(probs_vec, transform_mat, jac_full, T_prev + jnp.absolute(T), bundle_mask)
            loss_tuple = (loss, jnp.sum(jnp.absolute(T)), loss + reg * (jnp.sum(jnp.absolute(T)) - T_budget)**2,
                            reg * (jnp.sum(jnp.absolute(T)) - T_budget)**2)
            print(loss_tuple)
            losses += [loss_tuple]
        # print(time.time() - start_verbose)

    return np.array(losses), T

def fisher_sampling_1elec(probs_empirical, T_prev, amps, w_inits_array=None, t_final=None, 
                          budget=10000, reg=None, T_step_size=0.05, T_n_steps=2000, ms=[1],
                          verbose=True, R2_cutoff=-np.inf, return_probs=False,
                          min_prob=0.2, trial_cap=25,
                          exploit_factor=0.75, zero_prob=0.01, slope_bound=20, NUM_THREADS=24,
                          bootstrapping=None, X_all=None, reg_method='l2', regfit=[0],
                          R2_thresh=0.05, opt_verbose=False, bundle_mask=None):

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

    if bundle_mask is None:
        bundle_mask = np.ones(T_prev.shape, dtype=bool)
    else:
        assert bundle_mask.shape == T_prev.shape, "bundle_mask must be the same shape as T_prev"
        assert bundle_mask.dtype == bool, "bundle_mask must be a boolean array"

    # Create the array of all initial guesses if none is passed in
    if w_inits_array is None:
        w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
        for i in range(len(w_inits_array)):
            for j in range(len(w_inits_array[i])):
                w_inits = []

                for m in ms:
                    w_init = np.array(np.random.normal(size=(m, amps[j].shape[1]+1)))
                    z = 1 - (1 - zero_prob)**(1/len(w_init))
                    w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                    w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                    w_inits.append(w_init)

                w_inits_array[i][j] = w_inits

    print('Generating input list...')

    # Set up the data for multiprocess fitting
    input_list = fitting.generate_input_list(probs_empirical, amps, T_prev, w_inits_array, min_prob,
                                             bootstrapping=bootstrapping,
                                            X_all=X_all, zero_prob=zero_prob, slope_bound=slope_bound,
                                            reg_method=reg_method, reg=regfit, R2_thresh=R2_thresh,
                                            opt_verbose=opt_verbose)
    print('Fitting dataset...')

    params_curr = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    R2s = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]))
    probs_curr = np.zeros(probs_empirical.shape)

    if type(NUM_THREADS) == int:
        pool = mp.Pool(processes=NUM_THREADS)
        results = pool.starmap_async(fitting.fit_surface, input_list)
        mp_output = results.get()
        pool.close()

        cnt = 0
        for i in range(len(probs_empirical)):
            for j in range(len(probs_empirical[i])):
                params_curr[i][j] = mp_output[cnt][0][0]
                w_inits_array[i][j] = mp_output[cnt][1]
                R2s[i][j] = mp_output[cnt][0][2]
                
                probs_curr[i][j] = fitting.sigmoidND_nonlinear(
                                        sm.add_constant(amps[j], has_constant='add'), 
                                        params_curr[i][j])

                cnt += 1
    else:
        cnt = 0
        for i in range(len(probs_empirical)):
            for j in range(len(probs_empirical[i])):
                opt, w_inits_array[i][j] = fitting.fit_surface(*input_list[cnt])
                params_curr[i][j] = opt[0]
                R2s[i][j] = opt[2]
                probs_curr[i][j] = fitting.sigmoidND_nonlinear(
                                        sm.add_constant(amps[j], has_constant='add'), 
                                        params_curr[i][j])

                cnt += 1


    print('Calculating Jacobian...')

    jac_dict = collections.defaultdict(dict)
    transform_mat = []
    probs_vec = []
    num_params = 0

    for i in range(len(params_curr)):
        for j in range(len(params_curr[i])):
            if ~np.all(params_curr[i][j][:, 0] == -np.inf) and R2s[i][j] >= R2_cutoff:
                X = jnp.array(sm.add_constant(amps[j], has_constant='add'))
                jac_dict[i][j] = jax.jacfwd(activation_probs, argnums=1)(X, jnp.array(params_curr[i][j])).reshape(
                                                (len(X), params_curr[i][j].shape[0]*params_curr[i][j].shape[1]))  # c x l
                num_params += jac_dict[i][j].shape[1]

                transform = jnp.zeros(len(T_prev))
                transform = transform.at[j].set(1)
                transform_mat.append(transform)     # append a e-vector (512)

                probs_vec.append(probs_curr[i][j])  # append a c-vector (80)

    if len(probs_vec) == 0:
        raise ValueError("No valid probabilities found.")
    
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
        random_init = np.random.choice(len(T_prev.flatten()), size=int(budget*exploit_factor))
        T_new_init = jnp.array(np.bincount(random_init, minlength=len(T_prev.flatten())).astype(int).reshape(T_prev.shape), dtype='float32')

    else:
        T_new_init = jnp.array(jnp.absolute(jnp.array(t_final)), dtype='float32')

    losses, t_final = optimize_fisher_array(jac_full, probs_vec, transform_mat, jnp.array(T_prev, dtype='float32'), T_new_init, jnp.array(bundle_mask),
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

    T_new = np.array(T_new)
    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if np.sum(T_new) < budget:
        T_new_flat = T_new.flatten()
        bundle_mask_flat = bundle_mask.flatten()
        valid_indices = np.where(bundle_mask_flat)[0]

        random_extra = np.random.choice(valid_indices, size=int(budget - np.sum(T_new)), replace=True)
        counts = np.bincount(random_extra, minlength=len(T_new_flat))

        T_new_flat += counts
        T_new = T_new_flat.reshape(T_new.shape)

    capped_inds = np.where(T_new + T_prev >= trial_cap)
    T_new[capped_inds[0], capped_inds[1]] = np.clip(trial_cap - T_prev[capped_inds[0], capped_inds[1]],
                                                    0, None)

    if return_probs:
        return T_new.astype(int), w_inits_array, np.array(t_final), probs_curr, params_curr
    
    else:
        return T_new.astype(int), w_inits_array, np.array(t_final)