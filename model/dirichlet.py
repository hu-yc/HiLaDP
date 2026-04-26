import torch
import numpy as np
from torch import nn
from scipy.stats import beta
import torch.nn.functional as F

from model.gaussian import ReparametrizedGaussianVI
from settings import device


class DirichletProcess(nn.Module):
    def __init__(self, trunc, dim=1024, eta=10):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.device = device

        self.mu = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
        self.rho = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        self.gaussians = [ReparametrizedGaussianVI(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = torch.ones([self.dim, self.T]) / self.T

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=-1)

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).to(device)

        return samples

    def forward(self, x):
        batch_size = x.shape[0]

        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        entropy = self.entropy()
        entropy = entropy.expand(batch_size, -1)

        phi_new, kl_gaussian = self.get_phi(torch.log(pi), entropy, log_pdfs)

        self.update_gamma()

        likelihood = phi_new * kl_gaussian
        likelihood = likelihood.sum(1).mean(0)

        self.phi = phi_new.data

        return likelihood

    def infer(self, x):
        """
        Get logit

        return: Logits with length T
        """

        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs
        assert not torch.isnan(logits).any()
        # logits = F.normalize(logits, dim=2)

        return logits

    def get_phi(self, log_pi, entropy, log_pdf):
        # maybe mention this in the paper we do this to improve numerical stability
        kl_gaussian = log_pdf + entropy
        kl_pi = log_pi

        N_t_gaussian = kl_gaussian.sum(0, keepdim=True)
        N_t_pi = kl_pi.sum(0, keepdim=True)
        mix = (N_t_pi / (N_t_gaussian + N_t_pi))

        kl = mix * kl_gaussian + (1 - mix) * kl_pi

        return kl.softmax(dim=1), mix * kl_gaussian

    def update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def standardize(self, x):
        x = (x - x.mean(1)) / x.std(1)
        return x

    def update_mean(self, phi):
        N = phi.sum(0)
        pass

    def update_variance(self):
        pass
