import os
os.environ["CUDA_VISIBLE_DEVICES"]= '2'
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
import multielec_src.fitting as fitting
import multielec_src.multielec_utils as mutils
import statsmodels.api as sm
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

def sample_spikes(spikes, t):
    # Important: assumes spikes is shuffled
    """
    Helper function to sample spikes from a Bernoulli distribution.

    Parameters:
    spikes (np.ndarray AMPLITUDES X trials): possibly jagged array of spiking trials for a given (cell, pattern)
    t (np.ndarray AMPLITUDES X 1): Number of trials across amplitudes for a given (cell, pattern)

    Returns:
    p_empirical_array (np.ndarray): Empirical probability of a spike across
                              amplitude for a given (cell, pattern)
    """
    assert len(spikes) == len(t), "Number of amplitudes does not match number of trials"
    t = np.array(t).astype(int)
    
    p_empirical = np.zeros(len(t))
    for i in range(len(t)):
        # If there are no trials, set the empirical probability to 0.5
        if t[i] == 0:
            p_empirical[i] = 0.5
        else:
            if t[i] <= len(spikes[i]):
                p_empirical[i] = np.mean(spikes[i][:t[i]])
            else:
                p_empirical[i] = np.mean(spikes[i])
        
    return p_empirical

def sample_spikes_array(all_spikes, trials):
    """
    Sample spikes across all cells and patterns using multiprocessing.

    Parameters:
    all_spikes (np.ndarray CELLS X 1): spikes for all cells and patterns
    trials (np.ndarray AMPLITUDES x 1): Number of trials

    Returns:
    p_empirical_array (np.ndarray CELLS X AMPLITUDES): Empirical probability of a spike across
                                    all cells and patterns
    """

    # Set up a list for multiprocessing
    probs_empirical_array = np.zeros((len(all_spikes), len(trials)))
    for i in range(len(all_spikes)):
        probs_empirical_array[i] = sample_spikes(all_spikes[i], trials)
    
    return probs_empirical_array

def global_selectivity(probs_2d):
    target_probs = np.max(probs_2d, axis=0)
    target_inds = np.argmax(probs_2d, axis=0)
    selectivity = np.zeros_like(target_probs)

    for i, ind in enumerate(target_inds):
        other_inds = np.setdiff1d(np.arange(len(probs_2d)), ind)
        selectivity[i] = target_probs[i] * (1 - np.amax(probs_2d[other_inds, i]))
    
    return selectivity

def logit_r_squared(y_true, y_pred):
    y_true = np.clip(y_true, 1e-2, 1-1e-2)
    y_pred = np.clip(y_pred, 1e-2, 1-1e-2)
    y_true_logit = np.log(y_true/(1-y_true))
    y_pred_logit = np.log(y_pred/(1-y_pred))
    ss_res = np.sum((y_true_logit - y_pred_logit) ** 2)
    ss_tot = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
    return 1 - (ss_res / ss_tot)

def logit_ccc(y_true, y_pred):
    y_true = np.clip(y_true, 1e-2, 1-1e-2)
    y_pred = np.clip(y_pred, 1e-2, 1-1e-2)
    y_true_logit = np.log(y_true/(1-y_true))
    y_pred_logit = np.log(y_pred/(1-y_pred))
    # Concordance correlation coefficient

    # Means
    y_true_mean = np.mean(y_true_logit)
    y_pred_mean = np.mean(y_pred_logit)

    # Variances
    y_true_var = np.var(y_true_logit)
    y_pred_var = np.var(y_pred_logit)

    # correlation coefficient
    corr_coef = np.corrcoef(y_true_logit, y_pred_logit)[0, 1]

    ccc = 2 * corr_coef * np.sqrt(y_true_var) * np.sqrt(y_pred_var) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc

dataset = '2020-09-29-2'
basename = f'/Volumes/Analysis/{dataset}/gsort'
ESTIM_ANALYSIS_BASE = '/Volumes/Lab/Users/praful/outputs/pp_out'
datarun = 'data008'
wnoise = 'kilosort_data006/data006'

# Load electrical data and g-sort data
outpath = os.path.join(basename, datarun, wnoise)
parameters = loadmat(os.path.join(outpath, 'parameters.mat'))

cells = parameters['cells'].flatten()
patterns = parameters['patterns'].flatten()
num_cells = len(cells)
num_patterns = max(patterns)
num_movies = parameters['movies'].flatten()[0]

all_trials = np.array(np.memmap(os.path.join(outpath, 'trial.dat'),mode='r',shape=(num_patterns, num_movies), dtype='int16'), dtype=int)
amps_gsort = mutils.get_stim_amps_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, datarun), 
                                        len(all_trials))

path = os.path.join(basename, datarun, wnoise)
file_list = os.listdir(path)

ms = [8]
zero_prob = 0.01
slope_bound = 100
R2_thresh = 0.025
reg_param = 0.5
method = 'L-BFGS-B'
reg_method = 'l2'

patterns = {}
for file in file_list:
    if file.endswith('.mat') and file.startswith('fit'):
        pattern_cell = file.split('.mat')[0].split('_')[-1]
        p, c = pattern_cell.replace('p', '').split('c')
        p = int(p)
        c = int(c)
        print(p, c)
        
        params = loadmat(os.path.join(path, file))['params_true']
        X = loadmat(os.path.join(path, file))['amps_fit']
        probs_fit = loadmat(os.path.join(path, file))['probs_fit'].flatten()
        
        # Find indices of rows in A corresponding to rows in B, preserving order
        indices = [np.where(np.all(amps_gsort == row, axis=1))[0][0] for row in X]
        remaining_inds = np.setdiff1d(np.arange(len(amps_gsort)), indices)
        amps_remaining = deepcopy(amps_gsort[remaining_inds])

        probs_flipped = np.zeros(len(amps_gsort))
        probs_flipped[indices] = probs_fit

        probs_remaining = fitting.sigmoidND_nonlinear(sm.add_constant(amps_remaining, has_constant='add'),
                                                    params)
        probs_flipped[remaining_inds] = np.where(probs_remaining > 0.5, 1, 0)
        
        num_trials = all_trials[p-1]
        spikes_cp = []
        for i in range(len(num_trials)):
            num1s = int(np.around(probs_flipped[i] * num_trials[i], 0))
            num0s = num_trials[i] - num1s

            spikes_amp = np.random.permutation(np.concatenate((np.ones(num1s), np.zeros(num0s)))).astype(int)
            spikes_cp.append(spikes_amp)
        spikes_cp = np.array(spikes_cp, dtype=object)

        if p not in patterns:
            patterns[p] = []
        patterns[p].append((c, probs_flipped, spikes_cp, num_trials))

selectivities = {}
for p in patterns:
    cells = [c for c, _, _, _ in patterns[p]]
    probs_all = np.vstack([probs for _, probs, _, _ in patterns[p]])
    spikes_all = np.array([spikes for _, _, spikes, _ in patterns[p]], dtype=object)
    trials_all = patterns[p][0][-1]
    print(probs_all.shape, spikes_all.shape)

    if len(cells) < 2:
        continue

    selectivity = global_selectivity(probs_all)
    selectivity = np.clip(selectivity, 1e-2, 1-1e-2)

    selectivity_logit = np.log(selectivity/(1-selectivity))
    selectivities[p] = (selectivity, selectivity_logit, probs_all, spikes_all, trials_all)
    print(p, np.amax(selectivity))

for p in patterns:
    spikes_all = selectivities[p][3]
    probs_all = selectivities[p][2]

    for i in range(len(spikes_all)):
        for j in range(len(spikes_all[i])):
            assert np.around(np.mean(spikes_all[i][j]), 3) == np.around(probs_all[i][j], 3), f"Mismatch between spikes and true probabilities at ({i}, {j})"

beta = 2
lcb_cutoff = 3    # ln(0.8/0.2) = 1.5, ln(0.9/0.1) = 2.2
batch_size = 1000
num_steps = 10
init_trials = 1000

pattern = 311

NUM_RUNS = 100
success_fractions_all = []
success_fractions_random_all = []
num_samples_all = []
num_samples_random_all = []
RMSEs_all = []
RMSEs_all_logit = []
RMSEs_random_all = []
RMSEs_random_all_logit = []
RMSEs_multisite_all = []
RMSEs_multisite_all_logit = []
MAEs_all = []
MAEs_all_logit = []
MAEs_random_all = []
MAEs_random_all_logit = []
MAEs_multisite_all = []
MAEs_multisite_all_logit = []
R2s_all = []
R2s_random_all = []
R2s_multisite_all = []
CCCs_all = []
CCCs_random_all = []
CCCs_multisite_all = []

for run in range(NUM_RUNS):
    # Step 0: Initialize the data
    init_inds = np.random.choice(np.arange(len(amps_gsort), dtype=int), init_trials, replace=True)

    # Count occurrences of each index
    T_prev = np.bincount(init_inds, minlength=len(amps_gsort))
    T_prev_random = deepcopy(T_prev)
    max_trials = selectivities[pattern][4]

    subsample_inds = np.where(T_prev > 0)[0]
    probs = sample_spikes_array(selectivities[pattern][3], T_prev)
    selec_probs = global_selectivity(probs[:, subsample_inds])

    subsample_inds_random = deepcopy(subsample_inds)
    selec_probs_random = deepcopy(selec_probs)

    num_samples = []
    num_samples_random = []

    success_fractions = []
    success_fractions_random = []

    RMSEs = []
    RMSEs_logit = []
    MAEs = []
    MAEs_logit = []

    RMSEs_random = []
    RMSEs_random_logit = []
    MAEs_random = []
    MAEs_random_logit = []

    RMSEs_multisite = []
    RMSEs_multisite_logit = []
    MAEs_multisite = []
    MAEs_multisite_logit = []

    R2s = []
    R2s_random = []
    R2s_multisite = []
    CCCs = []
    CCCs_random = []
    CCCs_multisite = []
    for step in range(num_steps):
        # Step 1: fitting the GP model

        Xdata = amps_gsort[subsample_inds]
        selec_probs = np.clip(selec_probs, 1e-2, 1-1e-2)
        y = np.log(selec_probs/(1-selec_probs))

        Xdata_random = amps_gsort[subsample_inds_random]
        selec_probs_random = np.clip(selec_probs_random, 1e-2, 1-1e-2)
        y_random = np.log(selec_probs_random/(1-selec_probs_random))

        # Assuming Xdata and y are in numpy format, convert them to torch tensors
        amps_plot_torch = torch.tensor(amps_gsort, dtype=torch.float32)
        X_train = torch.tensor(Xdata, dtype=torch.float32)
        y_train = torch.tensor(y.reshape(-1), dtype=torch.float32)
        X_train_random = torch.tensor(Xdata_random, dtype=torch.float32)
        y_train_random = torch.tensor(y_random.reshape(-1), dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amps_plot_torch = amps_plot_torch.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_train_random = X_train_random.to(device)
        y_train_random = y_train_random.to(device)

        # Define the GP Model
        class GPRegressionModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
                self.mean_constant = ConstantMean()
                # self.mean_linear = LinearMean(input_size=train_x.size(1))
                self.covar_module = RBFKernel()

            def forward(self, x):
                # Add the constant and linear mean components manually
                mean_x = self.mean_constant(x)# + self.mean_linear(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Early stopping class
        class EarlyStopping:
            def __init__(self, patience=10, min_delta=0.0):
                """
                :param patience: How many epochs to wait before stopping when loss isn't decreasing.
                :param min_delta: Minimum change in monitored loss to qualify as an improvement.
                """
                self.patience = patience
                self.min_delta = min_delta
                self.counter = 0
                self.best_loss = None
                self.stop = False

            def step(self, val_loss):
                if self.best_loss is None:
                    self.best_loss = val_loss
                elif val_loss > self.best_loss - self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.stop = True
                else:
                    self.best_loss = val_loss
                    self.counter = 0

        # Initialize the likelihood and model
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(X_train, y_train, likelihood)

        model = model.to(device)
        likelihood = likelihood.to(device)

        # Set model and likelihood in training mode
        model.train()
        likelihood.train()

        # Use an optimizer
        optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=1e-2)

        # Set up marginal log likelihood for GPyTorch
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Early stopping setup
        early_stopping = EarlyStopping(patience=10, min_delta=1e-3)

        # Training loop with early stopping
        training_iter = 10000
        losses_train = []
        for i in range(training_iter):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output_train = model(X_train)

            loss_train = -mll(output_train, y_train)

            losses_train.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            early_stopping.step(loss_train.item())

            if early_stopping.stop:
                print(f"Early stopping triggered at iteration {i + 1}")
                break

            # if i % 10 == 0:
            #     print(f"Iteration {i + 1}/{training_iter} - Training Loss: {loss_train.item()}")
            #     print(f"  Lengthscale: {model.covar_module.lengthscale}")
            #     print(f"  Noise: {model.likelihood.noise_covar.noise}")

        # Initialize the likelihood and model
        likelihood_random = GaussianLikelihood()
        model_random = GPRegressionModel(X_train_random, y_train_random, likelihood_random)

        model_random = model_random.to(device)
        likelihood_random = likelihood_random.to(device)

        # Set model and likelihood in training mode
        model_random.train()
        likelihood_random.train()

        # Use an optimizer
        optimizer_random = torch.optim.AdamW([{'params': model_random.parameters()}], lr=1e-2)

        # Set up marginal log likelihood for GPyTorch
        mll_random = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_random, model_random)

        # Early stopping setup
        early_stopping_random = EarlyStopping(patience=10, min_delta=1e-3)

        # Training loop with early stopping
        training_iter = 10000
        losses_train_random = []
        for i in range(training_iter):
            model_random.train()
            likelihood_random.train()
            optimizer_random.zero_grad()
            output_train_random = model_random(X_train_random)

            loss_train_random = -mll_random(output_train_random, y_train_random)

            losses_train_random.append(loss_train_random.item())
            loss_train_random.backward()
            optimizer_random.step()
            early_stopping_random.step(loss_train_random.item())

            if early_stopping_random.stop:
                print(f"Early stopping triggered at iteration {i + 1}")
                break

            # if i % 10 == 0:
            #     print(f"Iteration {i + 1}/{training_iter} - Training Loss: {loss_train_random.item()}")
            #     print(f"  Lengthscale: {model_random.covar_module.lengthscale}")
            #     print(f"  Noise: {model_random.likelihood.noise_covar.noise}")

        # Step 2: Model evaluation and plotting

        # Model evaluation
        model.eval()
        model_random.eval()
        likelihood.eval()
        likelihood_random.eval()
        with torch.no_grad():
            # Get model predictions
            predictions = likelihood(model(amps_plot_torch))
            predictions_random = likelihood_random(model_random(amps_plot_torch))
            mean = predictions.mean
            mean_random = predictions_random.mean
            var = predictions.variance
            var_random = predictions_random.variance
            lower, upper = predictions.confidence_region()
            lower_random, upper_random = predictions_random.confidence_region()

        probs_multisite = np.zeros((len(selectivities[pattern][2]), len(amps_gsort)))
        for cell_idx in range(len(selectivities[pattern][2])):
            X_sub = deepcopy(amps_gsort[subsample_inds_random])
            probs_fit_sub = deepcopy(sample_spikes_array(selectivities[pattern][3], T_prev_random)[cell_idx, subsample_inds_random])
            T_sub = deepcopy(T_prev_random[subsample_inds_random])

            w_inits = []
            for m in ms:
                w_init = np.array(np.random.normal(size=(m, X_sub.shape[1]+1)))
                z = 1 - (1 - zero_prob)**(1/len(w_init))
                w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                w_inits.append(w_init)

            opt, _ = fitting.fit_surface_earlystop(X_sub, probs_fit_sub, T_sub, w_inits,
                                        reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                                        zero_prob=zero_prob, method=method,
                                        R2_thresh=R2_thresh, verbose=True                           
            )
            params_subsample, _, _ = opt
            probs_subsample = fitting.sigmoidND_nonlinear(sm.add_constant(amps_gsort, has_constant='add'), 
                                                                    params_subsample)
            probs_multisite[cell_idx, :] = probs_subsample

        selec_multisite = global_selectivity(probs_multisite)
        selec_multisite = np.clip(selec_multisite, 1e-2, 1-1e-2)
        selec_multisite_logit = np.log(selec_multisite/(1-selec_multisite))
        probs_pred = 1/(1 + np.exp(-mean.cpu().numpy().flatten()))
        probs_pred_random = 1/(1 + np.exp(-mean_random.cpu().numpy().flatten()))

        # Inds where either the prediction or the true selectivity is above the cutoff
        selective_inds = np.where((selectivities[pattern][1] > lcb_cutoff) | (mean.cpu().numpy().flatten() > lcb_cutoff))[0]
        selective_inds_random = np.where((selectivities[pattern][1] > lcb_cutoff) | (mean_random.cpu().numpy().flatten() > lcb_cutoff))[0]
        selective_inds_multisite = np.where((selectivities[pattern][1] > lcb_cutoff) | (selec_multisite_logit > lcb_cutoff))[0]

        RMSE = np.sqrt(np.mean((probs_pred[selective_inds] - selectivities[pattern][0][selective_inds])**2))
        RMSE_logit = np.sqrt(np.mean((mean.cpu().numpy().flatten()[selective_inds] - selectivities[pattern][1][selective_inds])**2))
        RMSE_random = np.sqrt(np.mean((probs_pred_random[selective_inds_random] - selectivities[pattern][0][selective_inds_random])**2))
        RMSE_random_logit = np.sqrt(np.mean((mean_random.cpu().numpy().flatten()[selective_inds_random] - selectivities[pattern][1][selective_inds_random])**2))
        RMSE_multisite = np.sqrt(np.mean((selec_multisite[selective_inds_multisite] - selectivities[pattern][0][selective_inds_multisite])**2))
        RMSE_multisite_logit = np.sqrt(np.mean((selec_multisite_logit[selective_inds_multisite] - selectivities[pattern][1][selective_inds_multisite])**2))
        MAE = np.mean(np.abs(probs_pred[selective_inds] - selectivities[pattern][0][selective_inds]))
        MAE_logit = np.mean(np.abs(mean.cpu().numpy().flatten()[selective_inds] - selectivities[pattern][1][selective_inds]))
        MAE_random = np.mean(np.abs(probs_pred_random[selective_inds_random] - selectivities[pattern][0][selective_inds_random]))
        MAE_random_logit = np.mean(np.abs(mean_random.cpu().numpy().flatten()[selective_inds_random] - selectivities[pattern][1][selective_inds_random]))
        MAE_multisite = np.mean(np.abs(selec_multisite[selective_inds_multisite] - selectivities[pattern][0][selective_inds_multisite]))
        MAE_multisite_logit = np.mean(np.abs(selec_multisite_logit[selective_inds_multisite] - selectivities[pattern][1][selective_inds_multisite]))

        R2 = logit_r_squared(selectivities[pattern][0], probs_pred)
        R2_random = logit_r_squared(selectivities[pattern][0], probs_pred_random)
        R2_multisite = logit_r_squared(selectivities[pattern][0], selec_multisite)

        CCC = logit_ccc(selectivities[pattern][0], probs_pred)
        CCC_random = logit_ccc(selectivities[pattern][0], probs_pred_random)
        CCC_multisite = logit_ccc(selectivities[pattern][0], selec_multisite)

        RMSEs.append(RMSE)
        RMSEs_logit.append(RMSE_logit)
        RMSEs_random.append(RMSE_random)
        RMSEs_random_logit.append(RMSE_random_logit)
        RMSEs_multisite.append(RMSE_multisite)
        RMSEs_multisite_logit.append(RMSE_multisite_logit)
        MAEs.append(MAE)
        MAEs_logit.append(MAE_logit)
        MAEs_random.append(MAE_random)
        MAEs_random_logit.append(MAE_random_logit)
        MAEs_multisite.append(MAE_multisite)
        MAEs_multisite_logit.append(MAE_multisite_logit)
        R2s.append(R2)
        R2s_random.append(R2_random)
        R2s_multisite.append(R2_multisite)
        CCCs.append(CCC)
        CCCs_random.append(CCC_random)
        CCCs_multisite.append(CCC_multisite)
        print(f'RMSE: {RMSE}, RMSE (Random): {RMSE_random}, RMSE (Multisite): {RMSE_multisite}')
        print(f'MAE: {MAE}, MAE (Random): {MAE_random}, MAE (Multisite): {MAE_multisite}')

        print(f'RMSE (logit): {RMSE_logit}, RMSE (Random, logit): {RMSE_random_logit}, RMSE (Multisite, logit): {RMSE_multisite_logit}')
        print(f'MAE (logit): {MAE_logit}, MAE (Random, logit): {MAE_random_logit}, MAE (Multisite, logit): {MAE_multisite_logit}')

        print(f'R2: {R2}, R2 (Random): {R2_random}, R2 (Multisite): {R2_multisite}')
        print(f'CCC: {CCC}, CCC (Random): {CCC_random}, CCC (Multisite): {CCC_multisite}')

        ucb = mean.cpu().numpy().flatten() + beta*np.sqrt(var.cpu().numpy().flatten())
        lcb = mean.cpu().numpy().flatten() - beta*np.sqrt(var.cpu().numpy().flatten())

        # Fraction of high selectivity points found
        if len(np.where(selectivities[pattern][1] > lcb_cutoff)[0]) == 0:
            success_fraction = 0
            success_fraction_random = 0
        else:
            success_fraction = (len(np.where(np.isin(subsample_inds, np.where(selectivities[pattern][1] > lcb_cutoff)[0]))[0])/
                                len(np.where(selectivities[pattern][1] > lcb_cutoff)[0]))
            success_fraction_random = (len(np.where(np.isin(subsample_inds_random, np.where(selectivities[pattern][1] > lcb_cutoff)[0]))[0])/
                                len(np.where(selectivities[pattern][1] > lcb_cutoff)[0]))
            
        success_fractions.append(success_fraction)
        success_fractions_random.append(success_fraction_random)

        lcb_cutoff_inds = np.where(lcb > lcb_cutoff)[0]
        max_sampled_inds = np.where(T_prev >= max_trials)[0]
        restricted_inds = np.union1d(max_sampled_inds, lcb_cutoff_inds)
        allowed_inds = np.setdiff1d(np.arange(len(amps_gsort)), restricted_inds)

        num_samples.append(np.sum(T_prev))
        num_samples_random.append(np.sum(T_prev_random))

        if len(np.where(ucb[allowed_inds] > lcb_cutoff)[0]) == 0:
            print('No points above cutoff')
            break
        
        new_inds = np.random.choice(allowed_inds[np.where(ucb[allowed_inds] > lcb_cutoff)[0]], 
                                    batch_size, replace=True)
        T_new = np.bincount(new_inds, minlength=len(amps_gsort))
        T_prev = T_prev + T_new
        subsample_inds = np.where(T_prev > 0)[0]
        probs = sample_spikes_array(selectivities[pattern][3], T_prev)
        selec_probs = global_selectivity(probs[:, subsample_inds])

        restricted_inds_random = np.where(T_prev_random >= max_trials)[0]
        allowed_inds_random = np.setdiff1d(np.arange(len(amps_gsort)), restricted_inds_random)

        new_inds_random = np.random.choice(allowed_inds_random, batch_size, replace=True)
        T_new_random = np.bincount(new_inds_random, minlength=len(amps_gsort))
        T_prev_random = T_prev_random + T_new_random
        subsample_inds_random = np.where(T_prev_random > 0)[0]
        probs_random = sample_spikes_array(selectivities[pattern][3], T_prev_random)
        selec_probs_random = global_selectivity(probs_random[:, subsample_inds_random])

    success_fractions_all.append(success_fractions)
    success_fractions_random_all.append(success_fractions_random)
    num_samples_all.append(num_samples)
    num_samples_random_all.append(num_samples_random)
    RMSEs_all.append(RMSEs)
    RMSEs_all_logit.append(RMSEs_logit)
    RMSEs_random_all.append(RMSEs_random)
    RMSEs_random_all_logit.append(RMSEs_random_logit)
    RMSEs_multisite_all.append(RMSEs_multisite)
    RMSEs_multisite_all_logit.append(RMSEs_multisite_logit)
    MAEs_all.append(MAEs)
    MAEs_all_logit.append(MAEs_logit)
    MAEs_random_all.append(MAEs_random)
    MAEs_random_all_logit.append(MAEs_random_logit)
    MAEs_multisite_all.append(MAEs_multisite)
    MAEs_multisite_all_logit.append(MAEs_multisite_logit)
    R2s_all.append(R2s)
    R2s_random_all.append(R2s_random)
    R2s_multisite_all.append(R2s_multisite)
    CCCs_all.append(CCCs)
    CCCs_random_all.append(CCCs_random)
    CCCs_multisite_all.append(CCCs_multisite)

success_fractions_all = np.array(success_fractions_all, dtype=object)
success_fractions_random_all = np.array(success_fractions_random_all, dtype=object)
num_samples_all = np.array(num_samples_all, dtype=object)
num_samples_random_all = np.array(num_samples_random_all, dtype=object)
RMSEs_all = np.array(RMSEs_all, dtype=object)
RMSEs_all_logit = np.array(RMSEs_all_logit, dtype=object)
RMSEs_random_all = np.array(RMSEs_random_all, dtype=object)
RMSEs_random_all_logit = np.array(RMSEs_random_all_logit, dtype=object)
RMSEs_multisite_all = np.array(RMSEs_multisite_all, dtype=object)
RMSEs_multisite_all_logit = np.array(RMSEs_multisite_all_logit, dtype=object)
MAEs_all = np.array(MAEs_all, dtype=object)
MAEs_all_logit = np.array(MAEs_all_logit, dtype=object)
MAEs_random_all = np.array(MAEs_random_all, dtype=object)
MAEs_random_all_logit = np.array(MAEs_random_all_logit, dtype=object)
MAEs_multisite_all = np.array(MAEs_multisite_all, dtype=object)
MAEs_multisite_all_logit = np.array(MAEs_multisite_all_logit, dtype=object)
R2s_all = np.array(R2s_all, dtype=object)
R2s_random_all = np.array(R2s_random_all, dtype=object)
R2s_multisite_all = np.array(R2s_multisite_all, dtype=object)
CCCs_all = np.array(CCCs_all, dtype=object)
CCCs_random_all = np.array(CCCs_random_all, dtype=object)
CCCs_multisite_all = np.array(CCCs_multisite_all, dtype=object)

np.savez(f'gp_lse_global_selectivity_{dataset}_p{pattern}-multisite-data-trials-fixedm-R2-logit-beta{beta}-L{int(lcb_cutoff)}.npz',
        success_fractions_all=success_fractions_all,
        success_fractions_random_all=success_fractions_random_all,
        num_samples_all=num_samples_all,
        num_samples_random_all=num_samples_random_all,
        RMSEs_all=RMSEs_all,
        RMSEs_all_logit=RMSEs_all_logit,
        RMSEs_random_all=RMSEs_random_all,
        RMSEs_random_all_logit=RMSEs_random_all_logit,
        RMSEs_multisite_all=RMSEs_multisite_all,
        RMSEs_multisite_all_logit=RMSEs_multisite_all_logit,
        MAEs_all=MAEs_all,
        MAEs_all_logit=MAEs_all_logit,
        MAEs_random_all=MAEs_random_all,
        MAEs_random_all_logit=MAEs_random_all_logit,
        MAEs_multisite_all=MAEs_multisite_all,
        MAEs_multisite_all_logit=MAEs_multisite_all_logit,
        R2s_all=R2s_all,
        R2s_random_all=R2s_random_all,
        R2s_multisite_all=R2s_multisite_all,
        CCCs_all=CCCs_all,
        CCCs_random_all=CCCs_random_all,
        CCCs_multisite_all=CCCs_multisite_all,
        beta=beta,
        lcb_cutoff=lcb_cutoff,
        )