from .abstract import EFDistribution, Node
from numpy import log


class Lognormal(EFDistribution):

    def __init__(self, K):

        super(Lognormal, self).__init__()
        self.K = K
        self.f = 0
        self.g = 0
        self.natural_statistic = [lambda x: log(x), lambda x: log(x)**2]

    def compute_g(self):
        self.g = 0

    def compute_f(self):
        self.f = 0

    def compute_u(self, x):
        self.u = x.copy()

    def convert_to_natural_parameter(self, p):
        return log(p)
