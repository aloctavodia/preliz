#!/usr/bin/env python3
"""
Script to generate distribution template files for PreliZ.
"""

import sys


def generate_distribution_template(class_name, parameters):
    """Generate a distribution template file."""
    mod_name = class_name.lower()
    param_names = ", ".join(parameters)
    param_self = ", ".join([f"self.{param}" for param in parameters])

    template = f'''"""
{class_name} distribution.
"""

import pytensor.tensor as pt
from distributions import {mod_name} as {mod_name}_

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, any_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.special import  cdf_bounds, ppf_bounds_cont


class {class_name}(Distribution):
    r"""
    {class_name} distribution.

    The pdf of this distribution is

    .. math::

        f(x \\mid {param_names}) = \\text{{TODO: Add PDF formula}}

    .. plot::
        :context: close-figs

        from preliz import {class_name}, style
        style.use('preliz-doc')
        p0s = [TODO: Add parameter values]
        p1s = [TODO: Add parameter values]
        for p0, p1 in zip(p0s, p1s):
            {class_name}(p0, p1).plot_pdf()

    ========  ==============================================================
    Support   TODO: Add support information
    Mean      TODO: Add mean formula
    Variance  TODO: Add variance formula  
    ========  ==============================================================

    {class_name} distribution has X alternative parameterizations. In terms of ...

        The link between the X alternatives is given by

    .. math::

        equation_here

    Parameters
    ----------
{generate_param_docs(parameters)}
    """

    def __init__(self, {param_names}=None):
        super().__init__()
        self.support = (TODO_LOWER_BOUND, TODO_UPPER_BOUND)
        self._parametrization({param_names})

    def _parametrization(self, {param_names}):
        #Handle alternative parameterization if any

{generate_parametrization(parameters)}
        self.param_names = {param_names}
        if True:  # Change
            self._update({param_names})

    def _update(self, {param_names}):
{generate_update(parameters)}


    def _fit_moments(self, {param_names}):
        TODO: Implement method
        self._update()

    def _fit_mle(self, sample):
        TODO: Implement method
        self._update()

    def pdf(self, x):
        return pyt_pdf(x, {param_self})

    def cdf(self, x):
        return pyt_cdf(x, {param_self})

    def ppf(self, q):
        return pyt_ppf(q, {param_self})

    def sf(self, x):
        return pyt_sf(x, {param_self})

    def isf(self, q):
        return pyt_isf(q, {param_self})

    def logpdf(self, x):
        return pyt_logpdf(x, {param_self})

    def logcdf(self, x):
        return pyt_logcdf(x, {param_self})

    def logsf(self, x):
        return pyt_logsf(x, {param_self})

    def logisf(self, q):
        return pyt_logisf(q, {param_self})

    def entropy(self):
        return pyt_entropy({param_self})

    def mean(self):
        return pyt_mean({param_self})

    def mode(self):
        return pyt_mode({param_self})

    def median(self):
        return pyt_median({param_self})

    def var(self):
        return pyt_var({param_self})

    def std(self):
        return pyt_std({param_self})

    def skewness(self):
        return pyt_skewness({param_self})

    def kurtosis(self):
        return pyt_kurtosis({param_self})

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        return pyt_rvs({param_self}, size=size, rng=random_state)


@pytensor_jit
def pyt_pdf(x, {param_names}):
    return {mod_name}_.pdf(x, {param_names})

@pytensor_jit
def pyt_cdf(x, {param_names}):
    return {mod_name}_.cdf(x, {param_names})

@pytensor_jit
def pyt_ppf(q, {param_names}):
    return {mod_name}_.ppf(q, {param_names})

@pytensor_jit
def pyt_sf(x, {param_names}):
    return {mod_name}_.sf(x, {param_names})

@pytensor_jit
def pyt_isf(q, {param_names}):
    return {mod_name}_.isf(q, {param_names})

@pytensor_jit
def pyt_logpdf(x, {param_names}):
    return {mod_name}_.logpdf(x, {param_names})  

@pytensor_jit
def pyt_logcdf(x, {param_names}):
    return {mod_name}_.logcdf(x, {param_names})

@pytensor_jit
def pyt_logsf(x, {param_names}):
    return {mod_name}_.logsf(x, {param_names})

@pytensor_jit
def pyt_logisf(q, {param_names}):
    return {mod_name}_.logisf(q, {param_names})

@pytensor_jit
def pyt_entropy({param_names}):
    return {mod_name}_.entropy({param_names})

@pytensor_jit
def pyt_mean({param_names}):
    return {mod_name}_.mean({param_names})

@pytensor_jit
def pyt_mode({param_names}):
    return {mod_name}_.mode({param_names})

@pytensor_jit
def pyt_median({param_names}):
    return {mod_name}_.median({param_names})

@pytensor_jit
def pyt_var({param_names}):
    return {mod_name}_.var({param_names})

@pytensor_jit
def pyt_std({param_names}):
    return {mod_name}_.std({param_names})

@pytensor_jit
def pyt_skewness({param_names}):
    return {mod_name}_.skewness({param_names})

@pytensor_jit
def pyt_kurtosis({param_names}):
    return {mod_name}_.kurtosis({param_names})

@pytensor_rng_jit
def pyt_rvs({param_names}, size, rng):
    return {mod_name}_.rvs({param_names}, size=size, random_state=rng)
'''

    return template


def generate_param_docs(parameters):
    """Generate parameter documentation."""
    docs = []
    for param in parameters:
        docs.append(f"    {param} : float\n        TODO: Describe {param} parameter")
    return "\n".join(docs)


def generate_parametrization(parameters):
    """Generate parameter updates."""
    return "\n".join([f"        self.{param} = {param}" for param in parameters])


def generate_update(parameters):
    update = "\n".join([f"        self.{param} = np.float64({param})" for param in parameters])
    update += (
        "\n        self.params = (" + ", ".join([f"self.{param}" for param in parameters]) + ")\n"
    )
    update += "        # Check if something else needed\n"
    update += "        self.is_frozen = True"
    return update


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_distribution.py <distribution_name> <param1> <param2> ...")
        print("Example: python generate_distribution.py Beta alpha beta")
        sys.exit(1)

    dist_name = sys.argv[1]
    parameters = sys.argv[2:]

    # Generate the template
    template_content = generate_distribution_template(dist_name, parameters)

    print(template_content)


if __name__ == "__main__":
    main()
