import numpy as np
import pytensor.tensor as pt
from distributions import normal as pynormal_

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)
from preliz.internal.special import mean_and_std


class PyNormal(Continuous):
    r"""
    Normal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma) =
           \frac{1}{\sigma \sqrt{2\pi}}
           \exp\left\{ -\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2 \right\}

    .. plot::
        :context: close-figs

        from preliz import Normal, style
        style.use('preliz-doc')
        mus = [0., 0., -2.]
        sigmas = [1, 0.5, 1]
        for mu, sigma in zip(mus, sigmas):
            Normal(mu, sigma).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\sigma^2`
    ========  ==========================================

    Normal distribution has 2 alternative parameterizations. In terms of mean and
    sigma (standard deviation), or mean and tau (precision).

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0).
    tau : float
        Precision (tau > 0).
    """

    def __init__(self, mu=None, sigma=None, tau=None):
        super().__init__()
        self.support = (-pt.inf, pt.inf)
        self._parametrization(mu, sigma, tau)

    def _parametrization(self, mu=None, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu and sigma, or mu and tau."
            )

        names = ("mu", "sigma")
        self.params_support = ((-pt.inf, pt.inf), (eps, pt.inf))

        if tau is not None:
            self.tau = tau
            sigma = from_precision(tau)
            names = ("mu", "tau")

        self.mu = mu
        self.sigma = sigma
        self.param_names = names
        if all_not_none(mu, sigma):
            self._update(mu, sigma)

    def _update(self, mu, sigma):
        self.mu = mu  # np.float64(mu)
        self.sigma = sigma  # np.float64(sigma)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        self._update(*mean_and_std(sample))

    def pdf(self, x):
        return pyt_pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return pyt_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        return pyt_ppf(q, self.mu, self.sigma)

    def sf(self, x):
        return pyt_sf(x, self.mu, self.sigma)

    def isf(self, q):
        return pyt_isf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return pyt_logpdf(x, self.mu, self.sigma)

    def logcdf(self, x):
        return pyt_logcdf(x, self.mu, self.sigma)

    def logsf(self, x):
        return pyt_logsf(x, self.mu, self.sigma)

    def logisf(self, q):
        return pyt_logisf(q, self.mu, self.sigma)

    def entropy(self):
        return pyt_entropy(self.mu, self.sigma)

    def mean(self):
        return pyt_mean(self.mu, self.sigma)

    def mode(self):
        return pyt_mode(self.mu, self.sigma)

    def median(self):
        return pyt_median(self.mu, self.sigma)

    def var(self):
        return pyt_var(self.mu, self.sigma)

    def std(self):
        return pyt_std(self.mu, self.sigma)

    def skewness(self):
        return pyt_skewness(self.mu, self.sigma)

    def kurtosis(self):
        return pyt_kurtosis(self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        return pyt_rvs(self.mu, self.sigma, size=size, rng=random_state)


@pytensor_jit
def pyt_pdf(x, mu, sigma):
    return pynormal_.pdf(x, mu, sigma)


@pytensor_jit
def pyt_cdf(x, mu, sigma):
    return pynormal_.cdf(x, mu, sigma)


@pytensor_jit
def pyt_ppf(q, mu, sigma):
    return pynormal_.ppf(q, mu, sigma)


@pytensor_jit
def pyt_sf(x, mu, sigma):
    return pynormal_.sf(x, mu, sigma)


@pytensor_jit
def pyt_isf(q, mu, sigma):
    return pynormal_.isf(q, mu, sigma)


@pytensor_jit
def pyt_logpdf(x, mu, sigma):
    return pynormal_.logpdf(x, mu, sigma)


@pytensor_jit
def pyt_logcdf(x, mu, sigma):
    return pynormal_.logcdf(x, mu, sigma)


@pytensor_jit
def pyt_logsf(x, mu, sigma):
    return pynormal_.logsf(x, mu, sigma)


@pytensor_jit
def pyt_logisf(q, mu, sigma):
    return pynormal_.logisf(q, mu, sigma)


@pytensor_jit
def pyt_entropy(mu, sigma):
    return pynormal_.entropy(mu, sigma)


@pytensor_jit
def pyt_mean(mu, sigma):
    return pynormal_.mean(mu, sigma)


@pytensor_jit
def pyt_mode(mu, sigma):
    return pynormal_.mode(mu, sigma)


@pytensor_jit
def pyt_median(mu, sigma):
    return pynormal_.median(mu, sigma)


@pytensor_jit
def pyt_var(mu, sigma):
    return pynormal_.var(mu, sigma)


@pytensor_jit
def pyt_std(mu, sigma):
    return pynormal_.std(mu, sigma)


@pytensor_jit
def pyt_skewness(mu, sigma):
    return pynormal_.skewness(mu, sigma)


@pytensor_jit
def pyt_kurtosis(mu, sigma):
    return pynormal_.kurtosis(mu, sigma)


@pytensor_rng_jit
def pyt_rvs(mu, sigma, size, rng):
    return pynormal_.rvs(mu, sigma, size=size, random_state=rng)
