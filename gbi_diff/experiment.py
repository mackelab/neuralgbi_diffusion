import logging
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from gbi_diff.dataset import SBIDataset
from gbi_diff.model.lit_module import SBI
from gbi_diff.utils.config import Config
from lightning.pytorch.callbacks import ModelCheckpoint


def train(config: Config, device: str = "cpu"):
    # TODO: fixup device config 
    if not torch.cuda.is_available() and "cuda" in device:
        logging.warning("cuda device was requested but not available. Fall back to cpu")
        device = "auto"
        accelerator = "cpu"

    if device.isnumeric():
        device = int(device)
        accelerator = "cuda"
    else:
        accelerator = device
    device = torch.device(device)

    #TODO: log config construct as hyperparameter for the whole pipeline
    trainer = Trainer(
        logger=(CSVLogger(config.results_dir), TensorBoardLogger(config.results_dir, log_graph=True)),
        precision=config.precision,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epochs,
        callbacks=ModelCheckpoint(
            config.results_dir, monitor="val_loss_epoch", save_top_k=3
        ),
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
