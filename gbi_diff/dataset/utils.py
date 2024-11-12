from sourcerer.simulators import TwoMoonsSimulator


# use standard parameters
def generate_moon(size: int):
    simulator = TwoMoonsSimulator()
    prior = simulator.sample_prior(size)
    likelihood_samples = simulator.sample(prior)
    return prior, likelihood_samples
