import os
os.environ["CUDA_VISIBLE_DEVICES"]= '2'
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
import multielec_src.fitting as fitting
import statsmodels.api as sm
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

def sample_spikes(p_true, t, error_rate_0=0, error_rate_1=0):
    """
    Helper function to sample spikes from a Bernoulli distribution.

    Parameters:
    p_true (np.ndarray AMPLITUDES X 1): True probabilities of spiking across amplitudes
                         for a given (cell, pattern)
    t (np.ndarray AMPLITUDES X 1): Number of trials across amplitudes for a given (cell, pattern)

    Returns:
    p_empirical_array (np.ndarray): Empirical probability of a spike across
                              amplitude for a given (cell, pattern)
    """
    p_true, t = np.array(p_true), np.array(t).astype(int)
    
    p_empirical = []
    for i in range(len(p_true)):
        # If there are no trials, set the empirical probability to 0.5
        if t[i] == 0:
            p_empirical += [0.5]
        
        # Else, sample from a Bernoulli distribution
        else:
            spikes = np.random.choice(np.array([0, 1]), 
                                                 p=np.array([(1-p_true[i])*(1-error_rate_0) + p_true[i]*error_rate_1, 
                                                             p_true[i]*(1-error_rate_1) + (1-p_true[i])*error_rate_0]), 
                                                 size=t[i])

            p_empirical += [np.mean(spikes)]
        
    p_empirical_array = np.array(p_empirical)

    return p_empirical_array

def global_selectivity(probs_2d):
    target_probs = np.max(probs_2d, axis=0)
    target_inds = np.argmax(probs_2d, axis=0)
    selectivity = np.zeros_like(target_probs)

    for i, ind in enumerate(target_inds):
        other_inds = np.setdiff1d(np.arange(len(probs_2d)), ind)
        selectivity[i] = target_probs[i] * (1 - np.amax(probs_2d[other_inds, i]))
    
    return selectivity

dataset = '2020-10-18-5'
basename = f'/Volumes/Analysis/{dataset}/gsort'
datarun = 'data006'
wnoise = 'kilosort_data002/data002'

amps_plot = np.array(np.meshgrid(np.linspace(-2, 2, 21), 
                                    np.linspace(-2, 2, 21),
                                    np.linspace(-2, 2, 21))).T.reshape(-1,3)

path = os.path.join(basename, datarun, wnoise)
file_list = os.listdir(path)

patterns = {}
for file in file_list:
    if file.endswith('.mat') and file.startswith('fit'):
        params = loadmat(os.path.join(path, file))['params_true']
        probs_pred = fitting.sigmoidND_nonlinear(sm.add_constant(amps_plot, has_constant='add'), 
                                                                params)

        pattern_cell = file.split('.mat')[0].split('_')[-1]
        p, c = pattern_cell.replace('p', '').split('c')
        p = int(p)
        c = int(c)
        if p not in patterns:
            patterns[p] = []
        patterns[p].append((c, probs_pred))

selectivities = {}
for p in patterns:
    cells = [c for c, _ in patterns[p]]
    probs_all = np.vstack([probs for _, probs in patterns[p]])
    print(probs_all.shape)

    selectivity = global_selectivity(probs_all)

    selectivity_logit = np.log(selectivity/(1-selectivity))
    selectivities[p] = (selectivity, selectivity_logit, probs_all)
    print(p, np.amax(selectivity))

beta = 2
lcb_cutoff = 1.5
batch_size = 100
num_steps = 30
init_fraction = 0.05
T = 15
error_rate_0 = 0.0
error_rate_1 = 0.0

pattern = 500

NUM_RUNS = 100
success_fractions_all = []
success_fractions_random_all = []
num_samples_all = []
num_samples_random_all = []
RMSEs_all = []
RMSEs_random_all = []
MAEs_all = []
MAEs_random_all = []

for run in range(NUM_RUNS):
    print(f'Run {run+1}/{NUM_RUNS}')

    # Step 0: Initialize the data

    subsample_inds = np.random.choice(np.arange(len(amps_plot)),
                                        size=int(init_fraction*len(amps_plot)),
                                        replace=False)
    subsample_inds_random = deepcopy(subsample_inds)

    probs = sample_spikes(selectivities[pattern][2][:, subsample_inds].flatten(), 
                        np.ones(len(subsample_inds)*len(selectivities[pattern][2]))*T,
                        error_rate_0=error_rate_0, error_rate_1=error_rate_1)
    selec_probs = global_selectivity(probs.reshape(len(selectivities[pattern][2]), len(subsample_inds)))
    selec_probs_random = deepcopy(selec_probs)

    num_samples = []
    num_samples_random = []

    success_fractions = []
    success_fractions_random = []

    RMSEs = []
    MAEs = []

    RMSEs_random = []
    MAEs_random = []
    for step in range(num_steps):
        # Step 1: fitting the GP model

        Xdata = amps_plot[subsample_inds]
        selec_probs = np.clip(selec_probs, 1e-2, 1-1e-2)
        y = np.log(selec_probs/(1-selec_probs))

        Xdata_random = amps_plot[subsample_inds_random]
        selec_probs_random = np.clip(selec_probs_random, 1e-2, 1-1e-2)
        y_random = np.log(selec_probs_random/(1-selec_probs_random))

        # Assuming Xdata and y are in numpy format, convert them to torch tensors
        amps_plot_torch = torch.tensor(amps_plot, dtype=torch.float32)
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
                self.covar_module = RBFKernel()

            def forward(self, x):
                mean_x = self.mean_constant(x)
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

            if i % 10 == 0:
                print(f"Iteration {i + 1}/{training_iter} - Training Loss: {loss_train.item()}")
                print(f"  Lengthscale: {model.covar_module.lengthscale}")
                print(f"  Noise: {model.likelihood.noise_covar.noise}")

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

            if i % 10 == 0:
                print(f"Iteration {i + 1}/{training_iter} - Training Loss: {loss_train_random.item()}")
                print(f"  Lengthscale: {model_random.covar_module.lengthscale}")
                print(f"  Noise: {model_random.likelihood.noise_covar.noise}")
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

        probs_pred = 1/(1 + np.exp(-mean.cpu().numpy().flatten()))
        probs_pred_random = 1/(1 + np.exp(-mean_random.cpu().numpy().flatten()))
        selective_inds = np.where(selectivities[pattern][1] > lcb_cutoff)[0]

        RMSE = np.sqrt(np.mean((probs_pred[selective_inds] - selectivities[pattern][0][selective_inds])**2))
        RMSE_random = np.sqrt(np.mean((probs_pred_random[selective_inds] - selectivities[pattern][0][selective_inds])**2))
        MAE = np.mean(np.abs(probs_pred[selective_inds] - selectivities[pattern][0][selective_inds]))
        MAE_random = np.mean(np.abs(probs_pred_random[selective_inds] - selectivities[pattern][0][selective_inds]))
        RMSEs.append(RMSE)
        RMSEs_random.append(RMSE_random)
        MAEs.append(MAE)
        MAEs_random.append(MAE_random)
        print(f'RMSE: {RMSE}, RMSE (Random): {RMSE_random}')
        print(f'MAE: {MAE}, MAE (Random): {MAE_random}')

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
        restricted_inds = np.union1d(subsample_inds, lcb_cutoff_inds)
        allowed_inds = np.setdiff1d(np.arange(len(amps_plot)), restricted_inds)

        num_samples.append(len(subsample_inds))
        num_samples_random.append(len(subsample_inds_random))

        if len(np.where(ucb[allowed_inds] > lcb_cutoff)[0]) == 0:
            print('No points above cutoff')
            break
        
        # new_inds = allowed_inds[np.flip(np.argsort(ucb[allowed_inds]))[:batch_size]]
        new_inds = np.random.choice(allowed_inds[np.where(ucb[allowed_inds] > lcb_cutoff)[0]], 
                                    size=batch_size if len(np.where(ucb[allowed_inds] > lcb_cutoff)[0]) >= batch_size else len(np.where(ucb[allowed_inds] > lcb_cutoff)[0]), 
                                    replace=False)
        subsample_inds = np.hstack((subsample_inds, new_inds))
        new_probs = sample_spikes(selectivities[pattern][2][:, new_inds].flatten(), 
                                np.ones(len(new_inds)*len(selectivities[pattern][2]))*T,
                                error_rate_0=error_rate_0, error_rate_1=error_rate_1)
        selec_probs = np.hstack((selec_probs, global_selectivity(new_probs.reshape(len(selectivities[pattern][2]), len(new_inds)))))

        allowed_inds_random = np.setdiff1d(np.arange(len(amps_plot)), subsample_inds_random)
        new_inds_random = np.random.choice(allowed_inds_random, size=len(new_inds), replace=False)
        subsample_inds_random = np.hstack((subsample_inds_random, new_inds_random))
        new_probs_random = sample_spikes(selectivities[pattern][2][:, new_inds_random].flatten(), 
                                        np.ones(len(new_inds_random)*len(selectivities[pattern][2]))*T,
                                        error_rate_0=error_rate_0, error_rate_1=error_rate_1)
        selec_probs_random = np.hstack((selec_probs_random, global_selectivity(new_probs_random.reshape(len(selectivities[pattern][2]), len(new_inds_random)))))
        
    success_fractions_all.append(success_fractions)
    success_fractions_random_all.append(success_fractions_random)
    num_samples_all.append(num_samples)
    num_samples_random_all.append(num_samples_random)
    RMSEs_all.append(RMSEs)
    RMSEs_random_all.append(RMSEs_random)
    MAEs_all.append(MAEs)
    MAEs_random_all.append(MAEs_random)

success_fractions_all = np.array(success_fractions_all, dtype=object)
success_fractions_random_all = np.array(success_fractions_random_all, dtype=object)
num_samples_all = np.array(num_samples_all, dtype=object)
num_samples_random_all = np.array(num_samples_random_all, dtype=object)
RMSEs_all = np.array(RMSEs_all, dtype=object)
RMSEs_random_all = np.array(RMSEs_random_all, dtype=object)
MAEs_all = np.array(MAEs_all, dtype=object)
MAEs_random_all = np.array(MAEs_random_all, dtype=object)

np.savez(f'gp_lse_global_selectivity_{dataset}_p{pattern}.npz',
        success_fractions_all=success_fractions_all,
        success_fractions_random_all=success_fractions_random_all,
        num_samples_all=num_samples_all,
        num_samples_random_all=num_samples_random_all,
        RMSEs_all=RMSEs_all,
        RMSEs_random_all=RMSEs_random_all,
        MAEs_all=MAEs_all,
        MAEs_random_all=MAEs_random_all)