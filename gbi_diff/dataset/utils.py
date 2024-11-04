from sourcerer.simulators import TwoMoonsSimulator


def generate_moon(size: int, mean_radius: float = 0.1, std_radius: float = 0.01):
    simulator = TwoMoonsSimulator(mean_radius=mean_radius, std_radius=std_radius)
    prior = simulator.sample_prior(size)
    posterior = simulator.sample(prior)
    return prior, posterior
