import logging
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from gbi_diff.dataset import SBIDataset
from gbi_diff.model.lit_module import SBI
from gbi_diff.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def train(config: Config, devices: int = 1):
    # TODO: fixup device config
    accelerator = "auto"
    if not torch.cuda.is_available() and devices > 1:
        logging.warning("cuda device was requested but not available. Fall back to cpu")
        devices = 1
        accelerator = "cpu"

    # TODO: log config construct as hyperparameter for the whole pipeline
    trainer = Trainer(
        default_root_dir=config.results_dir,
        logger=(
            # NOTE: make sure tensor board stays at first places
            TensorBoardLogger(config.results_dir, log_graph=True),
            CSVLogger(config.results_dir),
        ),
        precision=config.precision,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                config.results_dir,
                monitor="val/loss",
                save_top_k=3,
                mode="max",
            ),
            LearningRateMonitor("epoch"),
        ],
        accelerator=accelerator,
        devices=devices,
    )

    train_set = SBIDataset.from_file(config.train_file)
    train_set.set_n_target(config.n_target)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_worker,
    )
    val_set = SBIDataset.from_file(config.val_file)
    val_set.set_n_target(config.n_target)
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_worker,
    )
    model = SBI(
        prior_dim=train_set.get_prior_dim(),
        simulator_out_dim=train_set.get_sim_out_dim(),
        optimizer_config=config.optimizer.__dict__,
    )

    trainer.fit(model, train_loader, val_loader)

    model = SBI.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training
    # Test best model on validation and test set
    # val_result = trainer.test(model, datamodule=val_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
