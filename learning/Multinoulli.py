from .abstract import EFDistribution, Node
from numpy import log


class Multinoulli(EFDistribution):

    def __init__(self, K):

        super(Multinoulli, self).__init__()
        self.K = K
        self.f = 0
        self.g = 0

    def compute_g(self):
        self.g = 0

    def compute_f(self):
        self.f = 0

    def compute_u(self, x):
        self.u = x.copy()

    def convert_to_natural_parameter(self, p):
        return log(p)


class MultinoulliNode(Node):

    distribution = Multinoulli

    def __init__(self, K, **kwargs):
        self.dim = K
        super(Node, self).__init__(**kwargs)

