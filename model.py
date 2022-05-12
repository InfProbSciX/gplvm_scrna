
import torch
import gpytorch
import numpy as np
from tqdm import trange

from torch.distributions import Normal
from torch.distributions import kl_divergence

from gpytorch.mlls import VariationalELBO
from gpytorch.constraints import Interval
from gpytorch.models import ApproximateGP
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.variational import VariationalStrategy, \
    CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, LinearKernel, RBFKernel, \
    PeriodicKernel

softplus = torch.nn.Softplus()


class LatentVariable(gpytorch.Module):
    pass


class PointLatentVariable(LatentVariable):
    def __init__(self, X_init):
        super().__init__()
        self.register_parameter('X', torch.torch.nn.Parameter(X_init))

    def forward(self, batch_index=None, Y=None):
        return self.X[batch_index, :] if batch_index is not None \
               else self.X


class GPLVM(ApproximateGP):
    def __init__(self, n, data_dim, latent_dim, covariate_dim,
                 pseudotime_dim=True, n_inducing=60, period_scale=2*np.pi,
                 X_latent=None, X_covars=None):
        self.n = n
        self.q_l = latent_dim
        self.m = n_inducing
        self.q_c = covariate_dim
        self.q_p = pseudotime_dim
        self.batch_shape = torch.Size([data_dim])

        self.inducing_inputs = torch.randn(
            n_inducing, latent_dim + pseudotime_dim + covariate_dim)
        if pseudotime_dim:
            self.inducing_inputs[:, 0] = \
                torch.linspace(0, period_scale, n_inducing)

        q_u = CholeskyVariationalDistribution(n_inducing,
                                              batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs,
                                  q_u, learn_inducing_locations=False)

        super(GPLVM, self).__init__(q_f)

        self._init_gp_mean(covariate_dim)
        self._init_gp_covariance(
            data_dim, latent_dim, pseudotime_dim, covariate_dim, period_scale)
        self.X_latent = X_latent
        self.X_covars = X_covars

    def _init_gp_mean(self, covariate_dim):
        self.intercept = ConstantMean()
        if covariate_dim:
            self.random_effect_mean = LinearMean(covariate_dim, bias=False)
        else:
            self.random_effect_mean = ZeroMean()

    def _init_gp_covariance(self, d, q, pseudotime_dim, covariate_dim,
                            period_scale):

        self.pseudotime_dims = list(range(pseudotime_dim))
        if len(self.pseudotime_dims):
            period_length = Interval(period_scale-0.01, period_scale)
            pseudotime_covariance = PeriodicKernel(
                ard_num_dims=len(self.pseudotime_dims),
                active_dims=self.pseudotime_dims,
                period_length_constraint=period_length
            )
        else:
            pseudotime_covariance = None

        self.latent_var_dims = np.arange(pseudotime_dim, pseudotime_dim + q)
        if len(self.latent_var_dims):
            latent_covariance = RBFKernel(
                ard_num_dims=len(self.latent_var_dims),
                active_dims=self.latent_var_dims
            )
        else:
            latent_covariance = None

        max_dim = max(self.latent_var_dims, default=-1)
        max_dim = max(max_dim, max(self.pseudotime_dims, default=-1))
        self.known_var_dims = np.arange(covariate_dim + max_dim, max_dim, -1)
        self.known_var_dims.sort()
        if len(self.known_var_dims):
            random_effect_covariance = LinearKernel(
                ard_num_dims=len(self.known_var_dims),
                active_dims=self.known_var_dims
            )
        else:
            random_effect_covariance = None

        if not random_effect_covariance and not latent_covariance and \
           not pseudotime_covariance:
            raise ValueError('At least one covariance must be specified.')

        if pseudotime_covariance and latent_covariance:
            self.covar_module = pseudotime_covariance * latent_covariance
        elif pseudotime_covariance:
            self.covar_module = pseudotime_covariance
        elif latent_covariance:
            self.covar_module = latent_covariance
        else:
            self.covar_module = random_effect_covariance

        if (pseudotime_covariance or latent_covariance) and \
           random_effect_covariance:
            self.covar_module += random_effect_covariance

        self.covar_module = ScaleKernel(self.covar_module)
        # batch_shape=torch.Size([d])

    def forward(self, X):
        mean_x = self.intercept(X) + \
                 self.random_effect_mean(X[..., self.known_var_dims])
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist


class BatchIdx:
    def __init__(self, n, max_batch_size):
        self.n = n
        self.batch_size = max_batch_size
        self.indices = np.arange(n)
        np.random.shuffle(self.indices)

    def idx(self):
        min_idx = 0
        while True:
            min_idx_incr = min_idx + self.batch_size
            max_idx = min_idx_incr if (min_idx_incr <= self.n) else self.n
            yield self.indices[min_idx:max_idx]
            min_idx = 0 if min_idx_incr >= self.n else min_idx_incr


def train(gplvm, likelihood, Y, epochs=100, batch_size=100, lr=0.005):

    n = len(Y)
    steps = int(np.ceil(epochs*n/batch_size))
    elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
    optimizer = torch.optim.Adam([
        dict(params=gplvm.parameters(), lr=lr),
        dict(params=likelihood.parameters(), lr=lr)
    ])

    losses = []; idx = BatchIdx(n, batch_size).idx()
    iterator = trange(steps, leave=False)
    for i in iterator:
        batch_index = next(idx)
        optimizer.zero_grad()

        # ---------------------------------
        Y_batch = Y[batch_index]
        X_sample = torch.cat((
                gplvm.X_latent(batch_index, Y_batch),
                gplvm.X_covars[batch_index]
            ), axis=1)
        gplvm_dist = gplvm(X_sample)
        loss = -elbo_func(gplvm_dist, Y_batch.T).sum()
        # ---------------------------------

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    return losses


class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, data_dim, layers):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim
        self.prior_x = NormalPrior(
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self._init_nnet(layers)
        self.register_added_loss_term("x_kl")

        self.jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5

    def _init_nnet(self, hidden_layers):
        layers = (self.data_dim,) + hidden_layers + (self.latent_dim*2,)
        n_layers = len(layers)

        modules = []; last_layer = n_layers - 1
        for i in range(last_layer):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < last_layer - 1: modules.append(softplus)

        self.nnet = torch.nn.Sequential(*modules)

    def forward(self, batch_index=None, Y=None):
        h = self.nnet(Y)
        mu = h[..., :self.latent_dim].tanh()*5
        sg = softplus(h[..., self.latent_dim:]) + 1e-6

        q_x = torch.distributions.Normal(mu, sg)

        x_kl = _KL(q_x, self.prior_x, len(mu), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()


class _KL(AddedLossTerm):
    def __init__(self, q_x, p_x, n, d):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.d = d

    def loss(self):
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0)
        kl_per_point = kl_per_latent_dim.sum()/self.n
        return (kl_per_point/self.d)


__all__ = ['GPLVM', 'PointLatentVariable', 'NNEncoder', 'BatchIdx', 'train']
