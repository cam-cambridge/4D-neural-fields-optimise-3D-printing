#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import torch

torch.set_float32_matmul_precision(
    "medium"
)  # For performance with some precision trade-off

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import multiprocessing

# local dependencies
from src.utils.logger import Logger


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Interpol(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.max_epochs = self.config.training.epochs
        self.lr = self.config.training.learning_rate
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.epochs)  # T_max is the number of epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def loss_function(self, X, Y):
        
        Y_hat_recon = self.forward(X[:,:3], X[:,3:])
        reconstruction_loss = F.mse_loss(Y, Y_hat_recon)

        # lhs_samples = torch.tensor(lhs(4, samples=10000)).float().to('cuda')
        # space=  lhs_samples[:,:-1]
        # params= lhs_samples[:,-1:].requires_grad_(True)
        # Y_hat_regularization = self.forward(space, params)
        # dS_dp = torch.autograd.grad(Y_hat_regularization, params, create_graph=True, grad_outputs=torch.ones_like(Y_hat_regularization))[0]
        # regularization_term = torch.norm(dS_dp) ** 2
        
        # total_loss = reconstruction_loss+0.0001*regularization_term
        total_loss = reconstruction_loss

        # if self.training and self.global_rank == 0:
        #         self.logging_device.log("reconstruction", reconstruction_loss.item())
        #         self.logging_device.log("regularization", 0.0000001 * regularization_term.item())

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

    # ---- VALIDATION ----
    def on_validation_epoch_start(self):
        ...

    def validation_step(self, batch, batch_idx):

        torch.set_grad_enabled(True)

        input, output = batch
        
        loss = self.loss_function(input, output)

        torch.set_grad_enabled(False)



    def on_validation_epoch_end(self, unused=None):
        ...