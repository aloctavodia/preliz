"""
Beta distribution.
"""

import numpy as np
from distributions import beta as beta_

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    any_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import mean_and_std


class Beta(Continuous):
    r"""
    Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        from preliz import Beta, style
        style.use('preliz-doc')
        alphas = [.5, 5., 2.]
        betas = [.5, 5., 5.]
        for alpha, beta in zip(alphas, betas):
            ax = Beta(alpha, beta).plot_pdf()
        ax.set_ylim(0, 5)

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Beta distribution has 3 alternative parameterizations. In terms of alpha and
    beta, mean and sigma (standard deviation) or mean and nu (concentration).

    The link between the 3 alternatives is given by

    .. math::

       \alpha &= \mu \nu \\
       \beta  &= (1 - \mu) \nu

       \text{where } \nu = \frac{\mu(1-\mu)}{\sigma^2} - 1


    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    mu : float
        mean (0 < ``mu`` < 1).
    sigma : float
        standard deviation (``sigma`` < sqrt(``mu`` * (1 - ``mu``))).
    nu : float
        concentration > 0
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None, nu=None):
        super().__init__()
        self.support = (0, 1)
        self._parametrization(alpha, beta, mu, sigma, nu)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None, nu=None):
        if any_not_none(alpha, beta) and any_not_none(mu, sigma, nu) or all_not_none(sigma, nu):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta, or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            self.params_support = ((eps, 1 - eps), (eps, 1 - eps))
            if all_not_none(mu, sigma):
                alpha, beta = self._from_mu_sigma(mu, sigma)

        if any_not_none(mu, nu) and sigma is None:
            self.mu = mu
            self.nu = nu
            self.param_names = ("mu", "nu")
            self.params_support = ((eps, 1 - eps), (eps, np.inf))
            if all_not_none(mu, nu):
                alpha, beta = self._from_mu_nu(mu, nu)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

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

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)
        self.nu = self.mu * (1 - self.mu) / self.sigma**2 - 1

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "nu":
            self.params = (self.mu, self.nu)

        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)

    def pdf(self, x):
        return pyt_pdf(x, self.alpha, self.beta)

    def cdf(self, x):
        return pyt_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        return pyt_ppf(q, self.alpha, self.beta)

    def sf(self, x):
        return pyt_sf(x, self.alpha, self.beta)

    def isf(self, q):
        return pyt_isf(q, self.alpha, self.beta)

    def logpdf(self, x):
        return pyt_logpdf(x, self.alpha, self.beta)

    def logcdf(self, x):
        return pyt_logcdf(x, self.alpha, self.beta)

    def logsf(self, x):
        return pyt_logsf(x, self.alpha, self.beta)

    def logisf(self, q):
        return pyt_logisf(q, self.alpha, self.beta)

    def entropy(self):
        return pyt_entropy(self.alpha, self.beta)

    def mean(self):
        return pyt_mean(self.alpha, self.beta)

    def mode(self):
        return pyt_mode(self.alpha, self.beta)

    def median(self):
        return pyt_median(self.alpha, self.beta)

    def var(self):
        return pyt_var(self.alpha, self.beta)

    def std(self):
        return pyt_std(self.alpha, self.beta)

    def skewness(self):
        return pyt_skewness(self.alpha, self.beta)

    def kurtosis(self):
        return pyt_kurtosis(self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        return pyt_rvs(self.alpha, self.beta, size=size, rng=random_state)


@pytensor_jit
def pyt_pdf(x, alpha, beta):
    return beta_.pdf(x, alpha, beta)


@pytensor_jit
def pyt_cdf(x, alpha, beta):
    return beta_.cdf(x, alpha, beta)


@pytensor_jit
def pyt_ppf(q, alpha, beta):
    return beta_.ppf(q, alpha, beta)


@pytensor_jit
def pyt_sf(x, alpha, beta):
    return beta_.sf(x, alpha, beta)


@pytensor_jit
def pyt_isf(q, alpha, beta):
    return beta_.isf(q, alpha, beta)


@pytensor_jit
def pyt_logpdf(x, alpha, beta):
    return beta_.logpdf(x, alpha, beta)


@pytensor_jit
def pyt_logcdf(x, alpha, beta):
    return beta_.logcdf(x, alpha, beta)


@pytensor_jit
def pyt_logsf(x, alpha, beta):
    return beta_.logsf(x, alpha, beta)


@pytensor_jit
def pyt_logisf(q, alpha, beta):
    return beta_.logisf(q, alpha, beta)


@pytensor_jit
def pyt_entropy(alpha, beta):
    return beta_.entropy(alpha, beta)


@pytensor_jit
def pyt_mean(alpha, beta):
    return beta_.mean(alpha, beta)


@pytensor_jit
def pyt_mode(alpha, beta):
    return beta_.mode(alpha, beta)


@pytensor_jit
def pyt_median(alpha, beta):
    return beta_.median(alpha, beta)


@pytensor_jit
def pyt_var(alpha, beta):
    return beta_.var(alpha, beta)


@pytensor_jit
def pyt_std(alpha, beta):
    return beta_.std(alpha, beta)


@pytensor_jit
def pyt_skewness(alpha, beta):
    return beta_.skewness(alpha, beta)


@pytensor_jit
def pyt_kurtosis(alpha, beta):
    return beta_.kurtosis(alpha, beta)


@pytensor_rng_jit
def pyt_rvs(alpha, beta, size, rng):
    return beta_.rvs(alpha, beta, size=size, random_state=rng)
