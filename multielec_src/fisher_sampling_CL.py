import numpy as np
import matplotlib.pyplot as plt
import src.fitting as fitting
import statsmodels.api as sm
import jax
import jax.numpy as jnp
import multiprocessing as mp
import collections

def fisher_sampling_1elec(probs_empirical, T_prev, amps, w_inits_array=None, t_final=None, 
                          budget=10000, reg=20, T_step_size=0.01, T_n_steps=5000, ms=[1, 2],
                          verbose=True):

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

    input_list = fitting.generate_input_list(probs_empirical, amps, T_prev, w_inits_array)

    pool = mp.Pool(processes=24)
    results = pool.starmap_async(fitting.fit_surface, input_list)
    mp_output = results.get()
    pool.close()

    params_curr = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)
    w_inits_array = np.zeros((probs_empirical.shape[0], probs_empirical.shape[1]), dtype=object)

    probs_curr = np.zeros(probs_empirical.shape)
    cnt = 0
    for i in range(len(probs_empirical)):
        for j in range(len(probs_empirical[i])):
            params_curr[i][j] = mp_output[cnt][0]
            w_inits_array[i][j] = mp_output[cnt][1]
            
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
            if ~np.all(params_curr[i][j][:, 0] == -np.inf):
                X = jnp.array(sm.add_constant(amps[j], has_constant='add'))
                # jac_dict[i][j] = activation_probs_jac(X, jnp.array(params_curr[i][j]))
                jac_dict[i][j] = jax.jacfwd(fitting.activation_probs, argnums=1)(X, jnp.array(params_curr[i][j])).reshape(
                                                (len(X), params_curr[i][j].shape[0]*params_curr[i][j].shape[1]))  # c x l
                num_params += jac_dict[i][j].shape[1]

                transform = jnp.zeros(len(T_prev))
                transform = transform.at[j].set(1)
                transform_mat.append(transform)     # append a e-vector (512)

                probs_vec.append(probs_curr[i][j])  # append a c-vector (80)

    transform_mat = jnp.array(transform_mat)
    probs_vec = jnp.hstack(probs_vec)

    jac_full = jnp.zeros((len(probs_vec), num_params))
    counter_axis0 = 0
    counter_axis1 = 0
    for i in jac_dict.keys():
        for j in jac_dict[i].keys():
            next_jac = jac_dict[i][j]

            jac_full = jac_full.at[counter_axis0:counter_axis0+next_jac.shape[0], counter_axis1:counter_axis1+next_jac.shape[1]].set(next_jac)

            counter_axis0 += next_jac.shape[0]
            counter_axis1 += next_jac.shape[1]

    print('Optimizing trials...')

    if t_final is None:
        T_new_init = jnp.ones_like(jnp.array(T_prev), dtype=float)

    else:
        T_new_init = jnp.absolute(jnp.array(t_final))

    losses, t_final = fitting.optimize_fisher_array(jac_full, probs_vec, transform_mat, jnp.array(T_prev), T_new_init, 
                                                    step_size=T_step_size, n_steps=T_n_steps, reg=reg, T_budget=budget,
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
        plt.show()
        plt.savefig(f'plots_CL.png', dpi=300)

    T_new = jnp.round(jnp.absolute(t_final), 0)

    if jnp.sum(T_new) < budget:
        random_extra = np.random.choice(len(T_new.flatten()), size=int(budget - jnp.sum(T_new)),
                                        p=np.array(jnp.absolute(t_final.flatten()))/np.sum(np.array(jnp.absolute(t_final.flatten()))))
        T_new_extra = jnp.array(np.bincount(random_extra, minlength=len(T_new.flatten())).astype(int).reshape(T_new.shape))
        T_new = T_new + T_new_extra

    return np.array(T_new, dtype=int), w_inits_array, np.array(t_final)