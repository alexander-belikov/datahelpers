from numpy import log, arange, zeros
from numpy.testing import assert_array_almost_equal
from unittest import TestCase, main
import learning.Multinoulli as lm


class TestMulti(TestCase):

    K = 10
    m1 = lm.Multinoulli(K)
    probs = arange(1, K+1, dtype=float)/sum(arange(1, K+1))
    log_probs = log(probs)

    def test_init(self):
        self.assertTrue(self.m1.K == self.K)

    def test_logpdf(self):
        self.m1.set_phi(self.probs, True)
        x = zeros(self.K)
        i = 3
        x[i] = 1.0
        log_prob = self.m1.get_logpdf_given_phi(x)
        self.assertAlmostEqual(log_prob, log(self.probs[i]), delta=1e-12)

    def test_logpdf_vec(self):
        self.m1.set_phi(self.probs, True)
        n_obs = 7
        x = zeros((self.K, n_obs))
        for i in range(n_obs):
            x[i, i] = 1
        log_prob = self.m1.get_logpdf_given_phi(x)
        assert_array_almost_equal(log_prob, log(self.probs[:n_obs]), 8)


if __name__ == '__main__':
    main()
