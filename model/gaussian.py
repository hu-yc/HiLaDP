import math
import torch
from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class ReparametrizedGaussianVI(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        epsilon = epsilon.to(self.mean.device)
        return self.mean + self.std_dev * epsilon

    def log_prob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            # n_inputs, n_outputs = self.mean.shape
            dim = 1
            for d in self.mean.shape:
                dim *= d
        elif self.mean.dim() == 0:
            dim = 1
        else:
            dim = len(self.mean)
            # n_outputs = 1

        part1 = dim / 2 * (math.log(2 * math.pi) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return (part1 + part2).unsqueeze(0)

    def set_parameters(self, mu, rho):
        self.mean = mu
        self.rho = rho
