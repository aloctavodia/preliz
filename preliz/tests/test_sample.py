import numpy as np

import preliz as pz


def test_basic_normal():
    def model():
        return pz.Normal(0, 1)
    samples = pz.sample(model, size=1000)
    assert np.abs(samples.mean()) < 0.2
    assert np.abs(samples.std() - 1) < 0.2

def test_nested_truncated():
    def model():
        return pz.Truncated(pz.Normal(0, 1), lower=-1, upper=1)
    samples = pz.sample(model, size=1000)
    assert (samples >= -1).all() and (samples <= 1).all()

def test_hierarchical():
    def model():
        mu = pz.Normal(0, 1)
        return pz.Normal(mu, 1)
    samples = pz.sample(model, size=1000)
    assert np.abs(samples.mean()) < 0.3

def test_mixture():
    def model():
        return pz.Mixture([pz.Normal(0, 1), pz.Normal(5, 1)], weights=[0.5, 0.5])
    samples = pz.sample(model, size=1000)
    assert (np.abs(samples.mean()) < 3)

def test_composed():
    def model():
        prior = pz.Normal(176, 1)
        male_height = pz.Normal(mu=prior, sigma=7.1)
        female_height = pz.Truncated(pz.Normal(mu=prior, sigma=7.1), lower=150, upper=180)
        return (male_height > female_height)
    samples = pz.sample(model, size=1000)
    assert 0 <= samples.mean() <= 1

def test_no_rvs_on_imported():
    from preliz import Normal
    def model():
        return Normal(0, 1)
    # Should not add .rvs(), so returns a distribution object
    result = model()
    assert hasattr(result, 'rvs')
