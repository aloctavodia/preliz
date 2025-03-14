import numba as nb
import numpy as np
from scipy.special import expi, ndtr

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps
from preliz.internal.optimization import find_ppf, optimize_ml
from preliz.internal.special import cdf_bounds


class Wald(Continuous):
    r"""
    Wald distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::
        :context: close-figs


        from preliz import Wald, style
        style.use('preliz-doc')
        mus = [1., 1.]
        lams = [1., 3.]
        for mu, lam in zip(mus, lams):
            Wald(mu, lam).plot_pdf(support=(0,4))

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Wald distribution has 3 alternative parametrizations. In terms of mu and lam,
    mu and phi or lam and phi.

    The link between the 3 alternatives is given by

    .. math::

       \phi = \dfrac{\lambda}{\mu}

    Parameters
    ----------
    mu : float
        Mean of the distribution (mu > 0).
    lam : float
        Relative precision (lam > 0).
    phi : float
        Shape parameter (phi > 0).
    """

    def __init__(self, mu=None, lam=None, phi=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu, lam, phi)

    def _parametrization(self, mu=None, lam=None, phi=None):
        if all_not_none(mu, lam, phi):
            raise ValueError(
                "Incompatible parametrization. Either use mu and lam or mu and phi or lam and phi."
            )

        self.param_names = ("mu", "lam")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if phi is not None:
            self.phi = phi
            if (mu and phi) is not None:
                lam = self._from_mu_phi(mu, phi)
                self.param_names = ("mu", "phi")

            elif (lam and phi) is not None:
                mu = self._from_lam_phi(lam, phi)
                self.param_names = ("lam", "phi")

        self.mu = mu
        self.lam = lam
        if all_not_none(self.mu, self.lam):
            self._update(self.mu, self.lam)

    def _from_mu_phi(self, mu, phi):
        lam = mu * phi
        return lam

    def _from_lam_phi(self, lam, phi):
        mu = lam / phi
        return mu

    def _to_phi(self, mu, lam):
        phi = lam / mu
        return phi

    def _update(self, mu, lam):
        self.mu = np.float64(mu)
        self.lam = np.float64(lam)
        self.phi = self._to_phi(self.mu, self.lam)

        if self.param_names == ("mu", "lam"):
            self.params = (self.mu, self.lam)
        elif self.param_names == ("mu", "phi"):
            self.params = (self.mu, self.phi)
        elif self.param_names == ("lam", "phi"):
            self.params = (self.lam, self.phi)

        self.is_frozen = True

    def pdf(self, x):
        x = np.array(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.array(x)
        return nb_cdf(x, self.mu, self.lam)

    def ppf(self, q):
        q = np.asarray(q)
        return find_ppf(self, q)

    def logpdf(self, x):
        return nb_logpdf(x, self.mu, self.lam)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.mu, self.lam)

    def entropy(self):
        return nb_entropy(self.mu, self.lam)

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu * (
            (1 + (9 * self.mu**2) / (4 * self.lam**2)) ** 0.5 - (3 * self.mu) / (2 * self.lam)
        )

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.mu**3 / self.lam

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 3 * (self.mu / self.lam) ** 0.5

    def kurtosis(self):
        return 15 * self.mu / self.lam

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.wald(self.mu, self.lam, size)

    def _fit_moments(self, mean, sigma):
        lam = mean**3 / sigma**2
        self._update(mean, lam)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


def nb_cdf(x, mu, lam):
    x = np.asarray(x)
    u = (lam / (x + eps)) ** 0.5
    v = x / mu
    z = ndtr(u * (v - 1)) + np.exp(2 * lam / mu) * ndtr(-u * (v + 1))
    return cdf_bounds(z, x, 0, np.inf)


def nb_entropy(mu, lam):
    return 0.5 * np.log((2 * np.pi * np.e * mu**3) / lam) + 3 / 2 * np.exp(2 * lam / mu) * expi(
        -2 * lam / mu
    )


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, lam):
    if x <= 0:
        return -np.inf
    elif x == np.inf:
        return -np.inf
    else:
        return (
            np.log(lam) - (np.log(2 * np.pi) + 3 * np.log(x)) - lam * (x - mu) ** 2 / (mu**2 * x)
        ) / 2


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, lam):
    return -(nb_logpdf(x, mu, lam)).sum()
