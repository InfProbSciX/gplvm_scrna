
import sys, unittest
import torch, gpytorch
import numpy as np
import pickle as pkl
from tqdm import trange
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append('../')
from model import *
from model import _KL

torch.manual_seed(42); np.random.seed(42)
plt.ion(); plt.style.use('ggplot')

def _recurse_dims(kern):
    if hasattr(kern, 'base_kernel'):
        return _recurse_dims(kern.base_kernel)
    elif hasattr(kern, 'kernels'):
        dims_list = []
        for i in range(len(kern.kernels)):
            dims_list.append(_recurse_dims(kern.kernels[i]))
        dims_list = np.hstack(dims_list)
        dims_list.sort()
        return dims_list
    elif hasattr(kern, 'active_dims'):
        return np.array(kern.active_dims)
    else:
        raise RuntimeError

class GPLVMTests(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        n, q = 500, 2
        X = np.random.normal(size=(n, q))
        Y = np.vstack([
            0.1 * (X[:, 0] + X[:, 1])**2 - 3.5,
            0.01 * (X[:, 0] + X[:, 1])**3,
            2 * np.sin(0.5*(X[:, 0] + X[:, 1])),
            2 * np.cos(0.5*(X[:, 0] + X[:, 1])),
            4 - 0.1*(X[:, 0] + X[:, 1])**2,
            1 - 0.01*(X[:, 0] + X[:, 1])**3,
        ]).T
        d = len(Y.T)

        self.gplvm_data = n, d, q, X, Y

        X = np.random.normal(size=(n, q))
        beta_1 = np.random.normal(5, 1, size=n)
        beta_2 = np.random.normal(-2, 0.5, size=n)

        Y = np.vstack([
            beta_1*X[:, 0] + beta_2*X[:, 1],
            -beta_1*X[:, 0] - beta_2*X[:, 1],
        ]).T
        d = len(Y.T)

        self.re_data = n, d, q, X, Y

    def test_init_gplvm_no_x(self):
        with self.assertRaises(ValueError):
            model = GPLVM(n=1, data_dim=10, latent_dim=0, covariate_dim=0, pseudotime_dim=False)

    def _test_init(self, model):
        keys = ('n', 'batch_shape', 'm', 'q_p', 'q_l', 'q_c')
        n, d, m, q_p, q_l, q_c = [model.__getattr__(k) for k in keys]
        assert model.inducing_inputs.shape == torch.Size([m, q_l + q_p + q_c])
        assert model.variational_strategy.variational_distribution.loc.shape == torch.Size([d[0], m])

        re_mean = model.random_effect_mean(torch.ones(n, q_c)).detach()
        if q_c == 0: assert (re_mean == 0).all()
        assert len(re_mean) == n

        n_x_dims = q_l + q_p + q_c
        covar = model.covar_module(torch.ones(n, n_x_dims)).evaluate()
        if n_x_dims > 1:
            with self.assertRaises(RuntimeError):
                model.covar_module(torch.ones(n, n_x_dims - 1)).evaluate()

        active_dims = _recurse_dims(model.covar_module)
        np.testing.assert_allclose(active_dims, np.arange(n_x_dims))

    def test_inits_gplvm(self):
        cases = [ (2, 0, False), (0, 2, False), (0, 0, True),
                  (0, 1, True),  (2, 0, True),  (2, 3, True) ]

        for case in cases:
            q_l, q_c, q_p = case
            model = GPLVM(5, 3, latent_dim=q_l, covariate_dim=q_c, pseudotime_dim=q_p)
            self._test_init(model)

    def test_map_no_linear(self):
        n, d, q, X, Y = deepcopy(self.gplvm_data)
        Y = torch.tensor(Y)

        model = GPLVM(n, data_dim=d, latent_dim=q,
                      covariate_dim=0, pseudotime_dim=False, n_inducing=20)
        X_latent = PointLatentVariable(torch.randn(n, q))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        elbo_func = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=0.01),
            dict(params=likelihood.parameters(), lr=0.01),
            dict(params=X_latent.parameters(), lr=0.01)
        ])

        losses = []
        for i in range(1500):
            optimizer.zero_grad()
            loss = -elbo_func(model(X_latent()), Y.T).sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        Y_recon = model(X_latent()).loc.T.detach()
        assert (Y - Y_recon).abs().mean()/Y.std() < 0.0025

    def test_re_no_gplvm(self):
        n, d, q, X, Y = deepcopy(self.re_data)
        Y = torch.tensor(Y)
        X = torch.tensor(X).float()

        # with unittest.mock.patch.object(torch, 'randn', custom_randn):
        model = GPLVM(n, data_dim=d, latent_dim=0,
                      covariate_dim=2, pseudotime_dim=False, n_inducing=3)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        elbo_func = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=0.1),
            dict(params=likelihood.parameters(), lr=0.1),
        ])

        losses = []
        for i in range(500):
            optimizer.zero_grad()
            loss = -elbo_func(model(X), Y.T).sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        Y_recon = model(X).loc.T.detach()
        assert (Y - Y_recon).abs().mean() < 1

    def test_params_change(self):
        n, d, q, X, Y = deepcopy(self.gplvm_data)
        Y = torch.tensor(Y)
        X = torch.tensor(X).float()

        model = GPLVM(n, data_dim=d, latent_dim=q,
                      covariate_dim=q, pseudotime_dim=True, n_inducing=3)
        X_latent = PointLatentVariable(torch.randn(n, q+1))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        sd = {key: item.detach().clone() for key, item in model.named_parameters()}
        sd = {**sd, **{key: item.detach().clone() for key, item in X_latent.named_parameters()}}
        sd = {**sd, **{key: item.detach().clone() for key, item in likelihood.named_parameters()}}

        elbo_func = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=0.01),
            dict(params=likelihood.parameters(), lr=0.01),
            dict(params=X_latent.parameters(), lr=0.01)
        ])

        losses = []
        for i in range(10):
            optimizer.zero_grad()
            X_cat = torch.cat([X_latent(), X], axis=1)
            loss = -elbo_func(model(X_cat), Y.T).sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        sd_t = {key: item.detach().clone() for key, item in model.named_parameters()}
        sd_t = {**sd_t, **{key: item.detach().clone() for key, item in X_latent.named_parameters()}}
        sd_t = {**sd_t, **{key: item.detach().clone() for key, item in likelihood.named_parameters()}}

        assert min([(sd[key]-sd_t[key]).abs().max() for key in sd.keys()]) > 0

    def test_periodicity(self):
        for p_scale in (np.pi, 2*np.pi):
            model = GPLVM(5, 1, 0, 0, 1, 100, period_scale=p_scale)
            assert model.inducing_inputs[-1, 0] == p_scale
            sample = model(model.inducing_inputs).sample().detach()[0]
            assert (sample[0] - sample[-1]).abs()/sample.std() < 0.07
            assert model.covar_module.base_kernel.raw_period_length_constraint.upper_bound == p_scale

    def test_nnet_encoder_and_train(self):
        n, d, q, X, Y = deepcopy(self.gplvm_data)
        Y = torch.tensor(Y).float()

        model = GPLVM(n, data_dim=d, latent_dim=q,
                    covariate_dim=0, pseudotime_dim=False, n_inducing=20,
                    X_latent=NNEncoder(n, q, d, layers=(3, 2)),
                    X_covars=torch.zeros(n, 0))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        losses = train(gplvm=model, likelihood=likelihood,
              Y=Y, epochs=1000, batch_size=n, lr=0.01)

        Y_recon = model(model.X_latent(Y=Y)).loc.T.detach()

        assert (Y - Y_recon).abs().mean()/Y.std() < 0.02
        assert isinstance(next(model.added_loss_terms()), _KL)

    def test_nnet_encoder_mnist(self):
        from tensorflow.keras.datasets.mnist import load_data
        from scipy.spatial.distance import cdist
        torch.manual_seed(42); np.random.seed(42)

        n = 500
        (_, _), (Y, c) = load_data()
        Y = Y[:n]
        c = c[:n]
        Y = torch.tensor(Y.astype('f').reshape(-1, 28**2)/255)
        Y = Y + np.random.normal(size=Y.shape).astype('f')*0.01

        (n, d), (q, m) = Y.shape, (2, 64)

        model = GPLVM(n, data_dim=d, latent_dim=q,
                    covariate_dim=0, pseudotime_dim=False, n_inducing=20,
                    X_latent=NNEncoder(n, q, d, layers=(16, 8)),
                    X_covars=torch.zeros(n, 0))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if torch.cuda.is_available():
            Y = Y.cuda()
            model = model.cuda()
            model.X_covars = model.X_covars.cuda()
            likelihood = likelihood.cuda()

        losses = train(gplvm=model, likelihood=likelihood,
              Y=Y, epochs=400, batch_size=n, lr=0.01)

        losses = train(gplvm=model, likelihood=likelihood,
              Y=Y, epochs=4500, batch_size=n, lr=0.005)

        X = model.X_latent(Y=Y).detach().cpu()
        locs = np.vstack([X[c==lb].mean(axis=0) for lb in np.unique(c)])
        assert cdist(locs, locs)[np.tril_indices(10, -1)].mean() > 0.5

if __name__ == '__main__':
    unittest.main()
