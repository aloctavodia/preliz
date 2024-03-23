# pylint: disable=arguments-differ
import numpy as np

from preliz.distributions.distributions import TruncatedCensored
from preliz.internal.distribution_helper import all_not_none


class Truncated(TruncatedCensored):
    r"""
    Truncated distribution

    This is not a distribution per se, but a modifier of univariate distributions.

    The pdf of a Truncated distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \frac{\text{PDF}(x, dist)}{\text{CDF}(upper, dist) - \text{CDF}(lower, dist)}
            & \text{for } lower <= x <= upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gamma, Truncated
        az.style.use('arviz-doc')
        Truncated(Gamma(mu=2, sigma=1), 1, 4.5).plot_pdf()
        Gamma(mu=2, sigma=1).plot_pdf()
        

    Parameters
    ----------
    dist: PreliZ distribution
        Univariate PreliZ distribution which will be truncated.
    lower: float or int
        Lower (left) truncation point. Use np.inf for no truncation.
    upper: float or int
        Upper (right) truncation point. Use np.inf for no truncation.

    Note
    ----

    Some methods like mean or variance are not available truncated distributions.
    Functions like maxent or quantile are experimental when applied to  truncated
    distributions and may not work as expected.
    """

    def __init__(self, dist, lower=None, upper=None, **kwargs):
        self.dist = dist
        super().__init__()
        self._parametrization(lower, upper, **kwargs)

    def _parametrization(self, lower=None, upper=None, **kwargs):
        dist_params = []
        if not kwargs:
            if hasattr(self.dist, "params"):
                kwargs = dict(zip(self.dist.param_names, self.dist.params))
            else:
                kwargs = dict(zip(self.dist.param_names, [None] * len(self.dist.param_names)))

        for key, value in kwargs.items():
            dist_params.append(value)
            setattr(self, key, value)

        if upper is None:
            self.upper = np.inf
        else:
            self.upper = upper

        if lower is None:
            self.lower = -np.inf
        else:
            self.lower = lower

        self.params = (*dist_params, self.lower, self.upper)
        self.param_names = (*self.dist.param_names, "lower", "upper")
        if all_not_none(*dist_params):
            self.dist._parametrization(**kwargs)
            self.is_frozen = True

        self.support = (
            max(self.dist.support[0], self.lower),
            min(self.dist.support[1], self.upper),
        )
        self.params_support = (*self.dist.params_support, self.dist.support, self.dist.support)

    def median(self):
        return self.ppf(0.5)

    def rvs(self, size=1, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.uniform(size=size))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.asarray(x)
        lower = adjust_lower(self.kind, self.lower)
        lcdf = self.dist.cdf(lower)
        vals = (self.dist.cdf(x) - lcdf) / (self.dist.cdf(self.upper) - lcdf)
        return np.where(x < lower, 0, np.where(x > self.upper, 1, vals))

    def ppf(self, q):
        q = np.asarray(q)
        lower = adjust_lower(self.kind, self.lower)
        lcdf = self.dist.cdf(lower)
        vals = self.dist.ppf(lcdf + q * (self.dist.cdf(self.upper) - lcdf))
        return np.where((q < 0) | (q > 1), np.nan, vals)

    def logpdf(self, x):
        x = np.asarray(x)
        lower = adjust_lower(self.kind, self.lower)
        vals = self.dist.logpdf(x) - np.log(self.dist.cdf(self.upper) - self.dist.cdf(lower))
        return np.where((x < self.lower) | (x > self.upper), -np.inf, vals)

    def entropy(self):
        """
        This is the entropy of the UNtruncated distribution
        """
        if self.dist.rv_frozen is None:
            return self.dist.entropy()
        else:
            return self.dist.rv_frozen.entropy()

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def _fit_moments(self, mean, sigma):
        self.dist._fit_moments(mean, sigma)
        self._parametrization(**dict(zip(self.dist.param_names, self.dist.params)))


def adjust_lower(kind, lower):
    if kind == "discrete":
        lower -= 1
    return lower