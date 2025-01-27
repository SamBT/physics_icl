import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from model import GPT, GPTConfig
from PhysicsDatasets import DampedSHODatasetV2
import utils
import yaml
import os
import sys
from datetime import datetime

class DampedSHOTrainer(LightningModule):
    def __init__(self, config):
        super().__init__()
        # Load config
        self.config = config
        self.train_params = config['training_params']
        self.opt_params = config['opt_params']
        self.model_params = config['model_params']
        self.tokenized = self.model_params["tokenized"]
        
        # Define model
        gpt_config = GPTConfig(**self.model_params)
        self.model = GPT(gpt_config)
        
        # Loss function
        self.loss_fn = F.cross_entropy if self.tokenized else nn.MSELoss()
        self.tokenizer = None
        if self.tokenized:
            self.tokenizer = utils.RealNumberTokenizer(
                self.model_params['vocab_size'], self.train_params['range_limit_tok']
            )

    def forward(self, inpt, mask=None):
        return self.model(inpt, mask=mask)
    
    def training_step(self, batch, batch_idx):
        inpt, target, context, mask = batch
        if self.tokenized:
            inpt = self.tokenizer.tokenize(inpt).squeeze(-1)
            target = self.tokenizer.tokenize(target).squeeze(-1)
            mask = mask.squeeze(-1)
        
        if torch.all(mask == -999):
            mask = None
        if mask is not None:
            inpt_mask, tgt_mask = mask[:, :-1].float().to(self.device), mask[:, 1:].float().to(self.device)
        else:
            inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(self.device)

        preds = self(inpt.to(self.device), mask=inpt_mask)
        if self.tokenized:
            loss = self.loss_fn(
                preds.view(-1, preds.size(-1)),
                target.view(-1).to(self.device),
                reduction='none',
            )
            loss = (tgt_mask.view(-1) * loss).mean()
        else:
            loss = self.loss_fn(preds * tgt_mask, target.to(self.device) * tgt_mask)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inpt, target, context, mask = batch
        if self.tokenized:
            inpt = self.tokenizer.tokenize(inpt).squeeze(-1)
            target = self.tokenizer.tokenize(target).squeeze(-1)
            mask = mask.squeeze(-1)
        
        if torch.all(mask == -999):
            mask = None
        if mask is not None:
            inpt_mask, tgt_mask = mask[:, :-1].to(self.device), mask[:, 1:].float().to(self.device)
        else:
            inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(self.device)

        preds = self(inpt.to(self.device), mask=inpt_mask)
        if self.tokenized:
            loss = self.loss_fn(
                preds.view(-1, preds.size(-1)),
                target.view(-1).to(self.device),
                reduction='none',
            )
            loss = (tgt_mask.view(-1) * loss).mean()
        else:
            loss = self.loss_fn(preds * tgt_mask, target.to(self.device) * tgt_mask)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.opt_params['weight_decay'],
            self.opt_params['lr'],
            (self.opt_params['beta1'], self.opt_params['beta2']),
            device_type=self.device.type,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.opt_params['lr_decay_iter_frac'] * self.train_params['num_train_iters'],
                eta_min=self.opt_params['min_lr'],
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

def main(config_path):
    # Load configuration
    config = utils.load_config(config_path)
    seed_everything(42)

    # Dataset
    train_dataset = DampedSHODatasetV2(**config['dataset_params'])
    val_dataset = DampedSHODatasetV2(**config['dataset_params'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_params']['bs'],
        num_workers=1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training_params']['bs_val'],
        num_workers=1,
    )

    # Logger and Callbacks
    now = datetime.now().strftime("%d%b%y_%H%M")
    log_dir = f"logs/{config['model_name']}/{now}/"
    save_dir = f"trainings/{config['model_name']}/{now}/"
    os.makedirs(save_dir,exist_ok=True)
    logger = TensorBoardLogger("logs", name=config['model_name'], version=now)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_ckpt",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    regular_ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='ckpt_{step}',
        save_top_k=-1,
        every_n_train_steps=config['training_params']['save_every']
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = Trainer(
        max_steps=config['training_params']['num_train_iters'],
        max_epochs=int(config['training_params']['num_train_iters']/config['training_params']['val_every']),
        log_every_n_steps=10,
        val_check_interval=config['training_params']['val_every'],
        limit_val_batches=int(config['training_params']['num_val_seqs']/config['training_params']['bs_val'])+1,
        limit_train_batches=config['training_params']['val_every'],
        callbacks=[best_ckpt_callback, lr_monitor, regular_ckpt_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
    )

    # Model
    model = DampedSHOTrainer(config)

    # save config for later
    with open(f"{save_dir}/config.yaml","w") as fout:
        yaml.dump(config,fout)

    # Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        yamls = sys.argv[1:]
        for cfg in yamls:
            main(cfg)
    else:
        main(sys.argv[1])