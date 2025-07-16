import os
os.environ["CUDA_VISIBLE_DEVICES"]= ''
import numpy as np
from scipy.io import loadmat, savemat
import multielec_src.fitting as fitting
import multielec_src.multielec_utils as mutils
import statsmodels.api as sm
from copy import deepcopy
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

data = np.load('/Volumes/Lab/Users/praful/multielectrode/scripts/jupyter_notebooks/four_electrode_amps_and_spikes.npy')
print(data.shape)

# Randomly sample N points from the data
N = 104976
subsample_inds = np.random.choice(len(data), N, replace=False)

ms = [10]
zero_prob = 0.01
slope_bound = 100
reg_method = 'l2'
reg_param = 0.001
R2_thresh = 1e-7#0.025
method = 'L-BFGS-B'

w_inits = []
for m in ms:
    w_init = np.array(np.random.normal(size=(m, 5)))
    z = 1 - (1 - zero_prob)**(1/len(w_init))
    w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
    w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
    w_inits.append(w_init)

opt, _ = fitting.fit_surface_earlystop(data[subsample_inds, :4], data[subsample_inds, 4], np.ones(N), w_inits,
                            reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                            zero_prob=zero_prob, method=method,
                            R2_thresh=R2_thresh                           
)
params_true, _, R2 = opt
print(params_true)
probs_pred = fitting.sigmoidND_nonlinear(sm.add_constant(data[:, :4], has_constant='add'), 
                                                        params_true)

MAE_full = np.mean(np.abs(probs_pred - data[:, 4]))
RMSE_full = np.sqrt(np.mean((probs_pred - data[:, 4])**2))

print(f'MAE_full: {MAE_full}')
print(f'RMSE_full: {RMSE_full}')

# Step 1: fitting the GP model
Xdata_full = deepcopy(data[subsample_inds, :4])
probs_full = np.clip(data[subsample_inds, 4], 1e-2, 1-1e-2)
y_full = np.log(probs_full/(1-probs_full))

# Assuming Xdata and y are in numpy format, convert them to torch tensors
X_train_full = torch.tensor(Xdata_full, dtype=torch.float32)
y_train_full = torch.tensor(y_full.reshape(-1), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_full = X_train_full.to(device)
y_train_full = y_train_full.to(device)

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
likelihood_full = GaussianLikelihood()
model_full = GPRegressionModel(X_train_full, y_train_full, likelihood_full)

model_full = model_full.to(device)
likelihood_full = likelihood_full.to(device)

# Set model and likelihood in training mode
model_full.train()
likelihood_full.train()

# Use an optimizer
optimizer_full = torch.optim.AdamW([{'params': model_full.parameters()}], lr=1e-2)

# Set up marginal log likelihood for GPyTorch
mll_full = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_full, model_full)

# Early stopping setup
early_stopping_full = EarlyStopping(patience=10, min_delta=1e-3)

# Training loop with early stopping
training_iter = 10000
losses_train_full = []
for i in range(training_iter):
    model_full.train()
    likelihood_full.train()
    optimizer_full.zero_grad()
    output_train_full = model_full(X_train_full)

    loss_train_full = -mll_full(output_train_full, y_train_full)

    losses_train_full.append(loss_train_full.item())
    loss_train_full.backward()
    optimizer_full.step()
    early_stopping_full.step(loss_train_full.item())

    if early_stopping_full.stop:
        print(f"Early stopping triggered at iteration {i + 1}")
        break

    if i % 10 == 0:
        print(f"Iteration {i + 1}/{training_iter} - Training Loss: {loss_train_full.item()}")
        print(f"  Lengthscale: {model_full.covar_module.lengthscale}")
        print(f"  Noise: {model_full.likelihood.noise_covar.noise}")

# Model evaluation
model_full.eval()
likelihood_full.eval()

del optimizer_full
del mll_full
del output_train_full
del loss_train_full
del early_stopping_full
del X_train_full
del y_train_full

amps_plot_torch = torch.tensor(data[:, :4], dtype=torch.float32)
amps_plot_torch = amps_plot_torch.to(device)

batch_size = 5000
all_preds = []

with torch.no_grad():
    for i in range(0, len(amps_plot_torch), batch_size):
        print(i, len(amps_plot_torch))
        xb = amps_plot_torch[i:i+batch_size]
        preds = likelihood_full(model_full(xb))
        all_preds.append(preds.mean.cpu())  # move to CPU if you want to store it

mean_full = torch.cat(all_preds)

probs_pred_full = 1/(1 + np.exp(-mean_full.cpu().numpy().flatten()))

RMSE_full_gp = np.sqrt(np.mean((probs_pred_full - data[:, 4])**2))
MAE_full_gp = np.mean(np.abs(probs_pred_full - data[:, 4]))

print(f'RMSE (full, gp): {RMSE_full_gp}') 
print(f'MAE (full, gp): {MAE_full_gp}')

def logit_r_squared(y_true, y_pred):
    y_true = np.clip(y_true, 1e-2, 1-1e-2)
    y_pred = np.clip(y_pred, 1e-2, 1-1e-2)
    y_true_logit = np.log(y_true/(1-y_true))
    y_pred_logit = np.log(y_pred/(1-y_pred))
    ss_res = np.sum((y_true_logit - y_pred_logit) ** 2)
    ss_tot = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
    return 1 - (ss_res / ss_tot)

def log_likelihood(y_true, y_pred, trials):
    y_pred = np.clip(y_pred, 1e-2, 1-1e-2)
    LL = 0
    for i in range(len(y_true)):
        LL += trials[i] * y_true[i] * np.log(y_pred[i]) + trials[i] * (1 - y_true[i]) * np.log(1 - y_pred[i])

    return LL/np.sum(trials)

R2_full_multisite = logit_r_squared(data[:, 4], probs_pred)
R2_full_gp = logit_r_squared(data[:, 4], probs_pred_full)
print(f'R2_full_multisite: {R2_full_multisite}, R2_full_gp: {R2_full_gp}')

LL_full_multisite = log_likelihood(data[:, 4], probs_pred, np.ones(len(data)))
LL_full_gp = log_likelihood(data[:, 4], probs_pred_full, np.ones(len(data)))
LL_baseline = log_likelihood(data[:, 4], np.ones(len(data))*np.mean(data[:, 4]), np.ones(len(data)))
print(f'LL_full_multisite: {LL_full_multisite}, LL_full_gp: {LL_full_gp}, LL_baseline: {LL_baseline}')

MAE_baseline = np.mean(np.abs(data[:, 4] - np.mean(data[:, 4])))
RMSE_baseline = np.sqrt(np.mean((data[:, 4] - np.mean(data[:, 4]))**2))
print(f'MAE_baseline: {MAE_baseline}, RMSE_baseline: {RMSE_baseline}')

del model_full
del likelihood_full
del amps_plot_torch
del all_preds
del mean_full

torch.cuda.empty_cache()

NUM_RUNS = 100

RMSEs_all = []
MAEs_all = []
R2s_all = []
LLs_all = []

RMSEs_gp_all = []
MAEs_gp_all = []
R2s_gp_all = []
LLs_gp_all = []

stim_trials = [1000, 3000, 5000, 10000, 30000, 50000]
for run in range(NUM_RUNS):
    print(f'Run {run+1}/{NUM_RUNS}')

    RMSEs_subsample = []
    MAEs_subsample = []
    R2s_subsample = []
    LLs_subsample = []
    RMSEs_gp_subsample = []
    MAEs_gp_subsample = []
    R2s_gp_subsample = []
    LLs_gp_subsample = []
    for trial_count in stim_trials:
        subsample_inds = np.random.choice(len(data), trial_count, replace=False)

        w_inits = []
        for m in ms:
            w_init = np.array(np.random.normal(size=(m, 5)))
            z = 1 - (1 - zero_prob)**(1/len(w_init))
            w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
            w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
            w_inits.append(w_init)

        opt, _ = fitting.fit_surface_earlystop(data[subsample_inds, :4], data[subsample_inds, 4], np.ones(trial_count), w_inits,
                                    reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                                    zero_prob=zero_prob, method=method,
                                    R2_thresh=R2_thresh                           
        )
        params_true, _, R2 = opt
        probs_pred = fitting.sigmoidND_nonlinear(sm.add_constant(data[:, :4], has_constant='add'), 
                                                                params_true)

        MAE_subsample = np.mean(np.abs(probs_pred - data[:, 4]))
        RMSE_subsample = np.sqrt(np.mean((probs_pred - data[:, 4])**2))
        R2_subsample = logit_r_squared(data[:, 4], probs_pred)
        LL_subsample = log_likelihood(data[:, 4], probs_pred, np.ones(len(data)))

        RMSEs_subsample.append(RMSE_subsample)
        MAEs_subsample.append(MAE_subsample)
        R2s_subsample.append(R2_subsample)
        LLs_subsample.append(LL_subsample)

        # Step 1: fitting the GP model
        Xdata_subsample = deepcopy(data[subsample_inds, :4])
        probs_subsample = np.clip(data[subsample_inds, 4], 1e-2, 1-1e-2)
        y_subsample = np.log(probs_subsample/(1-probs_subsample))

        # Assuming Xdata and y are in numpy format, convert them to torch tensors
        X_train_subsample = torch.tensor(Xdata_subsample, dtype=torch.float32)
        y_train_subsample = torch.tensor(y_subsample.reshape(-1), dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_subsample = X_train_subsample.to(device)
        y_train_subsample = y_train_subsample.to(device)

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
        likelihood_subsample = GaussianLikelihood()
        model_subsample = GPRegressionModel(X_train_subsample, y_train_subsample, likelihood_subsample)

        model_subsample = model_subsample.to(device)
        likelihood_subsample = likelihood_subsample.to(device)

        # Set model and likelihood in training mode
        model_subsample.train()
        likelihood_subsample.train()

        # Use an optimizer
        optimizer_subsample = torch.optim.AdamW([{'params': model_subsample.parameters()}], lr=1e-2)

        # Set up marginal log likelihood for GPyTorch
        mll_subsample = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_subsample, model_subsample)

        # Early stopping setup
        early_stopping_subsample = EarlyStopping(patience=10, min_delta=1e-3)

        # Training loop with early stopping
        training_iter = 10000
        losses_train_subsample = []
        for i in range(training_iter):
            model_subsample.train()
            likelihood_subsample.train()
            optimizer_subsample.zero_grad()
            output_train_subsample = model_subsample(X_train_subsample)

            loss_train_subsample = -mll_subsample(output_train_subsample, y_train_subsample)

            losses_train_subsample.append(loss_train_subsample.item())
            loss_train_subsample.backward()
            optimizer_subsample.step()
            early_stopping_subsample.step(loss_train_subsample.item())

            if early_stopping_subsample.stop:
                break

        # Model evaluation
        model_subsample.eval()
        likelihood_subsample.eval()

        del optimizer_subsample
        del mll_subsample
        del output_train_subsample
        del loss_train_subsample
        del early_stopping_subsample
        del X_train_subsample
        del y_train_subsample

        amps_plot_torch = torch.tensor(data[:, :4], dtype=torch.float32)
        amps_plot_torch = amps_plot_torch.to(device)

        batch_size = 5000
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(amps_plot_torch), batch_size):
                xb = amps_plot_torch[i:i+batch_size]
                preds = likelihood_subsample(model_subsample(xb))
                all_preds.append(preds.mean.cpu())  # move to CPU if you want to store it

        mean_subsample = torch.cat(all_preds)

        probs_pred_subsample = 1/(1 + np.exp(-mean_subsample.cpu().numpy().flatten()))

        RMSE_gp = np.sqrt(np.mean((probs_pred_subsample - data[:, 4])**2))
        MAE_gp = np.mean(np.abs(probs_pred_subsample - data[:, 4]))
        R2_gp = logit_r_squared(data[:, 4], probs_pred_subsample)
        LL_gp = log_likelihood(data[:, 4], probs_pred_subsample, np.ones(len(data)))

        RMSEs_gp_subsample.append(RMSE_gp)
        MAEs_gp_subsample.append(MAE_gp)
        R2s_gp_subsample.append(R2_gp)
        LLs_gp_subsample.append(LL_gp)

        print(f'RMSE (gp): {RMSE_gp}') 
        print(f'MAE (gp): {MAE_gp}')
        print(f'R2 (gp): {R2_gp}')
        print(f'LL (gp): {LL_gp}')
        print(f'Stims: {trial_count}, RMSE: {RMSE_subsample}, MAE: {MAE_subsample}, R2: {R2_subsample}, LL: {LL_subsample}')

        del model_subsample
        del likelihood_subsample
        del amps_plot_torch
        del all_preds
        del mean_subsample

        torch.cuda.empty_cache()
    
    RMSEs_all.append(RMSEs_subsample)
    MAEs_all.append(MAEs_subsample)
    R2s_all.append(R2s_subsample)
    LLs_all.append(LLs_subsample)
    RMSEs_gp_all.append(RMSEs_gp_subsample)
    MAEs_gp_all.append(MAEs_gp_subsample)
    R2s_gp_all.append(R2s_gp_subsample)
    LLs_gp_all.append(LLs_gp_subsample)

RMSEs_all = np.array(RMSEs_all)
MAEs_all = np.array(MAEs_all)
R2s_all = np.array(R2s_all)
LLs_all = np.array(LLs_all)
RMSEs_gp_all = np.array(RMSEs_gp_all)
MAEs_gp_all = np.array(MAEs_gp_all)
R2s_gp_all = np.array(R2s_gp_all)
LLs_gp_all = np.array(LLs_gp_all)

savemat(os.path.join('/Volumes/Lab/Users/praful/multielectrode/figures', 'errors_4elec_subsample-gp-LL-fullrestarts.mat'), {
    'RMSEs_all': RMSEs_all,
    'MAEs_all': MAEs_all,
    'RMSEs_gp_all': RMSEs_gp_all,
    'MAEs_gp_all': MAEs_gp_all,
    'stim_trials': stim_trials,
    'RMSE_full': RMSE_full,
    'MAE_full': MAE_full,
    'RMSE_full_gp': RMSE_full_gp,
    'MAE_full_gp': MAE_full_gp,
    'R2s_all': R2s_all,
    'R2s_gp_all': R2s_gp_all,
    'R2_full': R2_full_multisite,
    'R2_full_gp': R2_full_gp,
    'LLs_all': LLs_all,
    'LLs_gp_all': LLs_gp_all,
    'LL_full': LL_full_multisite,
    'LL_full_gp': LL_full_gp,
    'LL_baseline': LL_baseline,
    'probs_raw': data[:, 4],
    'MAE_baseline': MAE_baseline,
    'RMSE_baseline': RMSE_baseline
})