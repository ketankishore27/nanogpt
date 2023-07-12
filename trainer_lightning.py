import os
from config.config import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from lightning_wrapper.data_module_wrapper import data_lightning_wrapper
from lightning_wrapper.gpt_model_wrapper import lightning_GPTModel_wrapper
from pytorch_lightning.loggers import WandbLogger

seed_everything(1331)
device = torch.device(device_comp)
wandb_logger = WandbLogger()
torch.set_float32_matmul_precision='medium'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def execute_flow():
    data = data_lightning_wrapper(path = "openwebtext", batch_size = batch_size)
    data.setup(stage='fit', make_files=False) # <-------------------------------------Check this before running
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # trainer = pl.Trainer(max_epochs = 60000, check_val_every_n_epoch=2, accelerator="mps", logger = wandb_logger,
    #                      gradient_clip_val=grad_clip)

    trainer = pl.Trainer(max_epochs = 60000, check_val_every_n_epoch=10, accelerator=device_comp, logger = wandb_logger,
                        gradient_clip_val=grad_clip,accumulate_grad_batches=40)
    model = lightning_GPTModel_wrapper()
#    model = torch.compile(model)
    checkpoint_path = None#input("Enter Checkpoint Name: ")
    if checkpoint_path:
        checkpoint_path = os.path.join('.', checkpoint_path)
        print("checkpoint path: {}".format(checkpoint_path))
        trainer.fit(model, train_dataloader, val_dataloader, 
                    ckpt_path = checkpoint_path)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    execute_flow()




