"""
BetaScaled distribution.
"""
import numpy as np
import pytensor.tensor as pt
from distributions import betascaled as betascaled_

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class BetaScaled(Continuous):
    r"""
    Scaled Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{(x-\text{lower})^{\alpha - 1} (\text{upper} - x)^{\beta - 1}}
           {(\text{upper}-\text{lower})^{\alpha+\beta-1} B(\alpha, \beta)}

    .. plot::
        :context: close-figs


        from preliz import BetaScaled, style
        style.use('preliz-doc')
        alphas = [2, 2]
        betas = [2, 5]
        lowers = [-0.5, -1]
        uppers = [1.5, 2]
        for alpha, beta, lower, upper in zip(alphas, betas, lowers, uppers):
            BetaScaled(alpha, beta, lower, upper).plot_pdf()

    ========  ==============================================================
    Support   :math:`x \in (lower, upper)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta} (upper-lower) + lower`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)} (upper-lower)`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, alpha=None, beta=None, lower=0, upper=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.support = (lower, upper)
        self._parametrization(self.alpha, self.beta, self.lower, self.upper)

    def _parametrization(self, alpha=None, beta=None, lower=None, upper=None):
        self.param_names = ("alpha", "beta", "lower", "upper")
        self.params_support = ((eps, pt.inf), (eps, pt.inf), (-pt.inf, pt.inf), (-pt.inf, pt.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta, lower, upper)

    def _update(self, alpha, beta, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta, self.lower, self.upper)
        self.support = self.lower, self.upper
        self.is_frozen = True

    def _from_mu_sigma(self, mu, sigma):
        nu = mu * (1 - mu) / sigma**2 - 1
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def _from_mu_nu(self, mu, nu):
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        alpha_plus_beta = alpha + beta
        mu = alpha / alpha_plus_beta
        sigma = (alpha * beta) ** 0.5 / alpha_plus_beta / (alpha_plus_beta + 1) ** 0.5
        return mu, sigma

    def _fit_moments(self, mean, sigma):
        mean = (mean - self.lower) / (self.upper - self.lower)
        sigma = sigma / (self.upper - self.lower)
        kappa = mean * (1 - mean) / sigma**2 - 1
        alpha = max(0.5, kappa * mean)
        beta = max(0.5, kappa * (1 - mean))
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        self._update(None, None, np.min(sample), np.max(sample))
        optimize_ml(self, sample)

    def pdf(self, x):
        return pyt_pdf(x, self.alpha, self.beta, self.lower, self.upper)

    def cdf(self, x):
        return pyt_cdf(x, self.alpha, self.beta, self.lower, self.upper)

    def ppf(self, q):
        return pyt_ppf(q, self.alpha, self.beta, self.lower, self.upper)

    def sf(self, x):
        return pyt_sf(x, self.alpha, self.beta, self.lower, self.upper)

    def isf(self, q):
        return pyt_isf(q, self.alpha, self.beta, self.lower, self.upper)

    def logpdf(self, x):
        return pyt_logpdf(x, self.alpha, self.beta, self.lower, self.upper)

    def logcdf(self, x):
        return pyt_logcdf(x, self.alpha, self.beta, self.lower, self.upper)

    def logsf(self, x):
        return pyt_logsf(x, self.alpha, self.beta, self.lower, self.upper)

    def logisf(self, q):
        return pyt_logisf(q, self.alpha, self.beta, self.lower, self.upper)

    def entropy(self):
        return pyt_entropy(self.alpha, self.beta, self.lower, self.upper)

    def mean(self):
        return pyt_mean(self.alpha, self.beta, self.lower, self.upper)

    def mode(self):
        return pyt_mode(self.alpha, self.beta, self.lower, self.upper)

    def median(self):
        return pyt_median(self.alpha, self.beta, self.lower, self.upper)

    def var(self):
        return pyt_var(self.alpha, self.beta, self.lower, self.upper)

    def std(self):
        return pyt_std(self.alpha, self.beta, self.lower, self.upper)

    def skewness(self):
        return pyt_skewness(self.alpha, self.beta, self.lower, self.upper)

    def kurtosis(self):
        return pyt_kurtosis(self.alpha, self.beta, self.lower, self.upper)

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        return pyt_rvs(self.alpha, self.beta, self.lower, self.upper, size=size, rng=random_state)


@pytensor_jit
def pyt_pdf(x, alpha, beta, lower, upper):
    return betascaled_.pdf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_cdf(x, alpha, beta, lower, upper):
    return betascaled_.cdf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_ppf(q, alpha, beta, lower, upper):
    return betascaled_.ppf(q, alpha, beta, lower, upper)


@pytensor_jit
def pyt_sf(x, alpha, beta, lower, upper):
    return betascaled_.sf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_isf(q, alpha, beta, lower, upper):
    return betascaled_.isf(q, alpha, beta, lower, upper)


@pytensor_jit
def pyt_logpdf(x, alpha, beta, lower, upper):
    return betascaled_.logpdf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_logcdf(x, alpha, beta, lower, upper):
    return betascaled_.logcdf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_logsf(x, alpha, beta, lower, upper):
    return betascaled_.logsf(x, alpha, beta, lower, upper)


@pytensor_jit
def pyt_logisf(q, alpha, beta, lower, upper):
    return betascaled_.logisf(q, alpha, beta, lower, upper)


@pytensor_jit
def pyt_entropy(alpha, beta, lower, upper):
    return betascaled_.entropy(alpha, beta, lower, upper)


@pytensor_jit
def pyt_mean(alpha, beta, lower, upper):
    return betascaled_.mean(alpha, beta, lower, upper)


@pytensor_jit
def pyt_mode(alpha, beta, lower, upper):
    return betascaled_.mode(alpha, beta, lower, upper)


@pytensor_jit
def pyt_median(alpha, beta, lower, upper):
    return betascaled_.median(alpha, beta, lower, upper)


@pytensor_jit
def pyt_var(alpha, beta, lower, upper):
    return betascaled_.var(alpha, beta, lower, upper)


@pytensor_jit
def pyt_std(alpha, beta, lower, upper):
    return betascaled_.std(alpha, beta, lower, upper)


@pytensor_jit
def pyt_skewness(alpha, beta, lower, upper):
    return betascaled_.skewness(alpha, beta, lower, upper)


@pytensor_jit
def pyt_kurtosis(alpha, beta, lower, upper):
    return betascaled_.kurtosis(alpha, beta, lower, upper)


@pytensor_rng_jit
def pyt_rvs(alpha, beta, lower, upper, size, rng):
    return betascaled_.rvs(alpha, beta, lower, upper, size=size, random_state=rng)
