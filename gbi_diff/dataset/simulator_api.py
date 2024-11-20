from sourcerer.simulators import (
    TwoMoonsSimulator,
    LotkaVolterraSimulator,
    InverseKinematicsSimulator,
)


# use standard parameters
def generate_two_moon(size: int):
    simulator = TwoMoonsSimulator()
    prior = simulator.sample_prior(size)
    likelihood_samples = simulator.sample(prior)
    return prior, likelihood_samples


def generate_lotka_volterra(size: int):
    simulator = LotkaVolterraSimulator()
    prior = simulator.sample_prior(size)
    likelihood_samples = simulator.sample(prior)
    return prior, likelihood_samples


def generate_inverse_kinematics(size: int):
    simulator = InverseKinematicsSimulator()
    prior = simulator.sample_prior(size)
    likelihood_samples = simulator.sample(prior)
    return prior, likelihood_samples
