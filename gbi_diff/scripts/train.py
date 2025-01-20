import logging
import sys

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from gbi_diff.dataset import dataset as sbi_datasets
from gbi_diff.model.lit_module import DiffusionModel, PotentialNetwork, Guidance
from gbi_diff.utils.cast import to_camel_case
from gbi_diff.utils.train_config import Config as Config_Potential
from gbi_diff.utils.train_guidance_config import Config as Config_Guidance
from gbi_diff.utils.train_diffusion_config import Config as Config_Diffusion


def _setup_trainer(
    config,
    devices: int = 1,
) -> Trainer:
    # TODO: fixup device config
    accelerator = "auto"
    if not torch.cuda.is_available() and devices > 1:
        logging.warning("cuda device was requested but not available. Fall back to cpu")
        devices = 1
        accelerator = "cpu"

    # setup logger
    tb_logger = TensorBoardLogger(config.results_dir, log_graph=True)
    csv_logger = CSVLogger(tb_logger.log_dir, name="csv_logs", version="")

    trainer = Trainer(
        default_root_dir=config.results_dir,
        logger=(
            # NOTE: make sure tensor board stays at first places
            tb_logger,
            csv_logger,
        ),
        precision=config.precision,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=tb_logger.log_dir,
                monitor="val/loss",
                save_top_k=3,
                mode="min",
            ),
            LearningRateMonitor("epoch"),
        ],
        accelerator=accelerator,
        devices=devices,
    )

    log_dir = tb_logger.log_dir
    return trainer, log_dir


def _setup_datasets(config):
    cls_name = to_camel_case(config.data_entity)
    cls_name = cls_name[0].upper() + cls_name[1:]
    dataset_cls = getattr(sbi_datasets, cls_name)
    train_set: sbi_datasets._SBIDataset = dataset_cls.from_file(config.dataset.train_file)
        
    train_set.set_n_target(config.dataset.n_target)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_worker,
    )
    val_set: sbi_datasets._SBIDataset = dataset_cls.from_file(config.dataset.val_file)
    val_set.set_n_target(config.dataset.n_target)
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_worker,
    )
    return train_loader, train_set, val_loader, val_set


def _print_state(config, model):
    print("============= Config ===============")
    print(yaml.dump(config.to_container(), indent=4))
    print("============== Net =================")
    print(model)


def _ask(force: bool):
    if force:
        return

    question = input("Would you like to start to train? [Y, n]")
    if not (question is None or question.lower().strip() in ["", "y", "yes"]):
        print("Abort training")
        sys.exit()


def train_potential(config: Config_Potential, devices: int = 1, force: bool = False):
    # TODO: fixup device config
    serial_config = config

    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)

    model = PotentialNetwork(
        theta_dim=train_set.get_theta_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
    )

    _print_state(config, model)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    trainer.fit(model, train_loader, val_loader)


def train_guidance(config: Config_Guidance, devices: int = 1, force: bool = False):
    # TODO: fixup device config
    serial_config = config
    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)
    
    model = Guidance(
        theta_dim=train_set.get_theta_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
        diff_config=config.diffusion,
    )
    _print_state(config, model)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    trainer.fit(model, train_loader, val_loader)


def train_diffusion(config: Config_Diffusion, devices: int = 1, force: bool = False):
    # TODO: fixup device config
    serial_config = config
    trainer, log_dir = _setup_trainer(config, devices)
    train_loader, train_set, val_loader, _ = _setup_datasets(config)

    model = DiffusionModel(
        theta_dim=train_set.get_theta_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
        diff_config=config.diffusion,
    )
    _print_state(config, model)
    _ask(force)

    serial_config.to_file(log_dir + "/config.yaml")
    trainer.fit(model, train_loader, val_loader)
