from gbi_diff.dataset.dataset import GaussianMixture
from gbi_diff.model.lit_module import DiffusionModel, Guidance, PotentialNetwork
from gbi_diff.model.utils import compute_multiplybymean_params, compute_standardizing_net_params
from gbi_diff.utils.configs.train_diffusion import Config as Config_Diffusion
from gbi_diff.utils.configs.train_guidance import Config as Config_Guidance
from gbi_diff.utils.configs.train_potential import Config as Config_Potential
from gbi_diff.utils.metrics import compute_distances, mmd_dist, mse_dist
from gbi_diff.utils.train_utils import (
    _ask,
    _print_state,
    _setup_datasets,
    _setup_trainer,
)


def train_potential(config: Config_Potential, devices: int = 1, force: bool = False):
    # TODO: fixup device config
    serial_config = config

    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)

    if isinstance(train_set, GaussianMixture):
        trial_dim = train_set._x.shape[1]
        dist_func = mmd_dist
    else:
        trial_dim = 0
        dist_func = mse_dist

    model = PotentialNetwork(
        theta_dim=train_set.get_theta_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
        trial_dim=trial_dim,
    )
    model.init_wrt_dataset(train_set._theta, train_set._x, train_set._x_miss, dist_func)

    _print_state(config, model, train_set)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    train_set.save_stats(log_dir)
    trainer.fit(model, train_loader, val_loader)


def train_guidance(config: Config_Guidance, devices: int = 1, force: bool = False):
    serial_config = config
    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)

    if isinstance(train_set, GaussianMixture):
        trial_dim = train_set._x.shape[1]
        dist_func = mmd_dist
    else:
        trial_dim = 0
        dist_func = mse_dist

    kwargs = {
        "theta_stats": compute_standardizing_net_params(train_set._theta, False),
        "x_stats": compute_standardizing_net_params(train_set._x, False),
        "distance_stats": compute_multiplybymean_params(dist_func, train_set._x_target, train_set._x),
    }
    model = Guidance(
        theta_dim=train_set.get_theta_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
        diff_config=config.diffusion,
        trial_dim=trial_dim,
        net_kwargs=kwargs       ,
    )
    # model.init_wrt_dataset(train_set._theta, train_set._x, train_set._x_miss, dist_func)

    _print_state(config, model)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    train_set.save_stats(log_dir)
    trainer.fit(model, train_loader, val_loader)


def train_diffusion(config: Config_Diffusion, devices: int = 1, force: bool = False):
    # TODO: fixup device config
    serial_config = config
    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)

    train_set.save_stats(log_dir)
    model = DiffusionModel(
        theta_dim=train_set.get_theta_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
        diff_config=config.diffusion,
    )
    _print_state(config, model, train_set)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    trainer.fit(model, train_loader, val_loader)
