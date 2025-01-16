#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pyDOE import lhs

# local dependencies
from src.utils import Logger
from src.sine import SineLayer

class Interpol(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.max_epochs = self.config.training.epochs
        self.lr = self.config.training.learning_rate
        self.gdir = self.config.gdir.gpus
        self.gpus = self.config.training.gpus

        self.net = []
        self.net.append(SineLayer(config.model.inputs, config.model.hidden, 
                                  is_first=True, omega_0= config.model.first_omega_0))

        for i in range(config.model.n_hidden):
            self.net.append(SineLayer(config.model.hidden, config.model.hidden, 
                                      is_first=False, omega_0=30.))

        if config.model.outermost_linear:
            final_linear = nn.Linear(config.model.hidden, config.model.outputs)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / config.model.hidden) / config.model.hidden_omega_0, 
                                              np.sqrt(6 / config.model.hidden) / config.model.hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(config.model.hidden, config.model.outputs, 
                                      is_first=False, omega_0= config.model.hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
        self.save_hyperparameters()

    def forward(self, space, params):
        input= torch.concat([space, params], axis=1)
        output = self.net(input)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Rprop(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.epochs)  # T_max is the number of epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def loss_function(self, X, Y):
        
        Y_hat_recon = self.forward(X[:,:3], X[:,3:])
        reconstruction_loss = F.mse_loss(Y, Y_hat_recon)

        regularization_term=torch.tensor(0)
        if self.gdir>0:
            lhs_samples = torch.tensor(lhs(4, samples=10000)).float().to('cuda')
            space=  lhs_samples[:,:-1]
            params= lhs_samples[:,-1:].requires_grad_(True)
            Y_hat_regularization = self.forward(space, params)
            dS_dp = torch.autograd.grad(Y_hat_regularization, params, create_graph=True, grad_outputs=torch.ones_like(Y_hat_regularization))[0]
            regularization_term = torch.norm(dS_dp) ** 2
        total_loss = reconstruction_loss + self.LAMBDA * regularization_term

        if self.training and self.global_rank == 0:
                self.logging_device.log("reconstruction", reconstruction_loss.item())
                self.logging_device.log("regularization", self.LAMBDA * regularization_term.item())

        self.log(
            "loss",
            total_loss*1000,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True,
        )

        return total_loss

    def on_fit_start(self):
        if self.global_rank==0:
            self.logging_device= Logger(
                batch=self.config.training.batch, 
                v_num=self.trainer.logger.version
            )

    def training_step(self, batch, batch_idx):
        input, output = batch
        loss = self.loss_function(input, output)

        if self.global_rank==0 and batch_idx%1==0 and batch_idx!=0:
            self.logging_device.report_running_mean(plot=False)
            
        return loss

    def on_train_epoch_end(self, unused=None):
        current_lr = self.optimizers().param_groups[0]["lr"]
        if self.global_rank == 0:
            print(f"Current Learning Rate (opt1): {current_lr}")