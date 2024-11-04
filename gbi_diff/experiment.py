from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from gbi_diff.dataset import SBIDataset
from gbi_diff.model.lit_module import SBI
from gbi_diff.utils.config import Config


def train(config: Config):
    trainer = Trainer(
        logger=(CSVLogger(config.results_dir), TensorBoardLogger(config.results_dir)),
        precision=config.precision,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epochs,
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
