import logging
import os

import torch
import yaml
import matplotlib
from config2class.utils import deconstruct_config
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from gbi_diff.dataset import SBIDataset
from gbi_diff.model.lit_module import SBI
from gbi_diff.utils.train_config import Config
from gbi_diff.utils.filesystem import write_yaml


def train(config: Config, devices: int = 1, force: bool = False):
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

    train_set = SBIDataset.from_file(config.dataset.train_file)
    train_set.set_n_target(config.dataset.n_target)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_worker,
    )
    val_set = SBIDataset.from_file(config.dataset.val_file)
    val_set.set_n_target(config.dataset.n_target)
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_worker,
    )
    model = SBI(
        prior_dim=train_set.get_prior_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer,
        net_config=config.model,
    )

    print("============= Config ===============")
    print(yaml.dump(deconstruct_config(config), indent=4))
    print("============== Net =================")
    print(model)

    if not force:
        question = input("Would you like to start to train? [Y, n]")
        if not (question is None or question.lower().strip() in ["", "y", "yes"]):
            print("Abort training")
            return

    write_yaml(deconstruct_config(config), tb_logger.log_dir + "/config.yaml")
    trainer.fit(model, train_loader, val_loader)

    # model = SBI.load_from_checkpoint(
    #     trainer.checkpoint_callback.best_model_path
    # )  # Load best checkpoint after training
    # Test best model on validation and test set
    # val_result = trainer.test(model, datamodule=val_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
