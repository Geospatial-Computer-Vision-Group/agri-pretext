import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner import Tuner
from utils import Normalize
from mae_model import mae_vit_small_patch16
from model import unpatchify
from data import SickleMAEPretrainDataModule

SEED = 42
L.seed_everything(SEED)
torch.set_float32_matmul_precision('medium')

class MAEModel(L.LightningModule):
    def __init__(self, learning_rate=1e-4, warmup_epochs=10, mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.model = mae_vit_small_patch16(norm_pix_loss=norm_pix_loss, img_size=224, in_chans=4)
        stats = torch.load("stats.pth")
        self.norm = Normalize(mean=stats["mean"], std=stats["std"])
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.automatic_optimization = True
        self.save_hyperparameters()
    
    def log_recon(self, actual, mask, pred, stage="train"):
        def minmax(x):
            x_min = x.amin(dim=(2,3),keepdim=True)
            x_max = x.amax(dim=(2,3),keepdim=True)
            return (x - x_min)/(x_max - x_min + 1e-9)
        pred = minmax((self.norm.denormalize(unpatchify(pred,16,4))).clamp_(0)[:,:3,...])# patch size, RGB image
        actual = minmax(self.norm.denormalize(actual))
        B,C,H,W = actual.shape
        h = w = int(mask.shape[1]**.5)
        p = self.model.patch_embed.patch_size[0]
        mask = mask.reshape(B,h,w).repeat_interleave(p,dim=1).repeat_interleave(p,dim=2)

        masked_actual = (actual * (1 - mask.unsqueeze(1).expand(-1, C, -1, -1)))[:,:3,...]
        masked_pred = pred * (1 - mask.unsqueeze(1).expand(-1, C, -1, -1)[:,:3,...])

        log_images = {
            f"{stage}_actual": actual[:4,:3],
            f"{stage}_masked_actual": masked_actual[:4],
            f"{stage}_masked_pred": masked_pred[:4],
            f"{stage}_recon": pred[:4,:3],
            f"{stage}_mask": mask.unsqueeze(1)[:4]
        }

        for tag, img in log_images.items():
            self.logger.experiment.add_images(tag, img, self.global_step)


    def training_step(self, batch, batch_idx):
        x = self.norm.normalize(batch["chip"])
        loss,pred,mask = self.model.forward(x,self.mask_ratio)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            with torch.no_grad():
                self.log_recon(x,mask,pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.norm.normalize(batch["chip"])
        loss,pred,mask = self.model.forward(x,self.mask_ratio)

        # Log validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            with torch.no_grad():
                self.log_recon(x,mask,pred,"val")
        return loss

    def configure_optimizers(self):
        params = self.model.parameters()

        optimizer = torch.optim.AdamW([
            {'params': params, 'lr': self.learning_rate},
        ])

        # Calculate total training steps and warmup steps
        max_epochs = self.trainer.max_epochs
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps / max_epochs
        warmup_steps = self.warmup_epochs * steps_per_epoch

        # Define the learning rate lambda function
        def lr_lambda(current_step: int):
            # Linear warmup for warmup_steps
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            # Cosine annealing after warmup
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # Create the scheduler with step-wise updates
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update per batch/step
                "frequency": 1,
            }
        }
    
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    cli = LightningCLI(
        MAEModel,
        SickleMAEPretrainDataModule,
        run=True,
    )
    # tuner = Tuner(cli.trainer)
    # lr_finder = tuner.lr_find(cli.model,datamodule=cli.datamodule)
    # suggested_lr = lr_finder.suggestion()
    # cli.trainer.model.learning_rate = suggested_lr
    # print(f"Suggested LR: {suggested_lr}")
    #cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    main()
