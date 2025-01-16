#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# local dependencies
from src.interpol import Interpol
from src.dataset.data_module import H5PYModule

def to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: to_namespace(v) for k, v in d.items()})
    return d

with open("config.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
    config = to_namespace(config_dict)

if __name__ == "__main__":

    data = H5PYModule(config=config)
    model = Interpol(config=config)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        filename=f"{config.name}_" + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
    )

    logger = CSVLogger("logs/train")

    trainer = pl.Trainer(
        deterministic=False,
        devices=config.training.gpus if config.training.gpus > 0 else 1,
        accelerator="gpu" if config.training.gpus > 0 else "cpu",
        max_epochs=config.training.epochs,
        strategy="ddp_find_unused_parameters_false",
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        limit_train_batches=1.0,
        limit_val_batches=0.0,
        precision="16-mixed",
        num_sanity_val_steps=0,  # Optional: skips sanity check to save time, can be removed
    )

    trainer.fit(model, data)
