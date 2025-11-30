import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from utils import Normalize
from model import FreqDecoder,patchify,unpatchify
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import SicklePretrainDataModule
import math
import gc

SEED = 42
L.seed_everything(SEED)
torch.set_float32_matmul_precision('medium')

class FreqPredModel(L.LightningModule):
    def __init__(self, model_name, stats_path, learning_rate=1e-4, warmup_epochs=10, img_size=224, patch_size=16, encoder_dim=384, dim=192, decoder_depth=6, decoder_heads=4, norm_pix_loss=True):
        super().__init__()
        # creating encoder
        self.encoder = timm.create_model(model_name,pretrained=False,img_size=img_size,patch_size=patch_size, in_chans=4) # hard-coded channels to 4 for R G B NIR
        del self.encoder.fc_norm
        del self.encoder.head_drop
        del self.encoder.head
        self.encoder.forward = self.encoder.forward_features
        self.encoder.__dict__.pop('forward_head', None)

        # freq decoder
        self.freq_decoder = FreqDecoder(encoder_dim,dim,img_size,patch_size,3,decoder_depth,decoder_heads) # top-3 freq channels

        stats = torch.load(stats_path,map_location=torch.device("cpu"))
        self.norm = Normalize(mean=stats["mean"], std=stats["std"]) # R G B NIR
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.norm_pix_loss = norm_pix_loss
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.decoder_depth = decoder_depth
        self.decoder_heads = decoder_heads
        self.img_size = img_size
        self.patch_size = patch_size
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward_encoder(self, images):
        x = self.encoder(images)[:,1:,...] # without cls token
        return x

    def forward_decoder(self,x1,x2,m1,m2,y1,y2):
        x = self.freq_decoder(x1,x2,m1,m2,y1,y2)
        return x
    
    def log_recon(self,pred,target,stage="train"):
        pred = (unpatchify(pred,16,3)).clamp_(0,1)# patch size, RGB image
        target = target.clamp_(0,1)
        # Assuming images are in a format that can be displayed (e.g., normalized between 0 and 1)
        self.logger.experiment.add_images(f'{stage}_actual', target[[0]], self.global_step)  # Log first image of batch
        self.logger.experiment.add_images(f'{stage}_recon', pred[[0]], self.global_step)
    
    def get_image(self, pred):
        pred = (unpatchify(pred,16,3)).clamp_(0,1)
        return pred

    def forward_loss(self,pred,target):
        target = patchify(target,self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        freq = batch["chip_freq"]
        x1 = x1[:,[2,1,0,3],...] # R G B NIR
        x2 = x2[:,[2,1,0,3],...]

        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        z = self.forward_encoder(torch.cat((x1,x2),dim=0))
        z1,z2 = z[:B],z[B:]
        pred = self.forward_decoder(z1,z2,m1,m2,y1,y2)

        loss = self.forward_loss(pred,freq)
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if batch_idx == 0:
            with torch.no_grad():
                self.log_recon(pred,freq)
        return loss

    def validation_step(self, batch, batch_idx):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        freq = batch["chip_freq"]
        x1 = x1[:,[2,1,0,3],...] # R G B NIR
        x2 = x2[:,[2,1,0,3],...]

        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        z = self.forward_encoder(torch.cat((x1,x2),dim=0))
        z1,z2 = z[:B],z[B:]
        pred = self.forward_decoder(z1,z2,m1,m2,y1,y2)

        loss = self.forward_loss(pred,freq)
        # Log validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if batch_idx == 0:
            with torch.no_grad():
                self.log_recon(pred,freq,"val")
        return loss
    
    def predict_step(self, batch):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        freq = batch["chip_freq"]
        x1 = x1[:,[2,1,0,3],...] # R G B NIR
        x2 = x2[:,[2,1,0,3],...]

        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        z = self.forward_encoder(torch.cat((x1,x2),dim=0))
        z1,z2 = z[:B],z[B:]
        pred = self.forward_decoder(z1,z2,m1,m2,y1,y2)

        return self.get_image(pred)

    def configure_optimizers(self):
        encoder_params = self.encoder.parameters()
        other_params = [p for n, p in self.named_parameters() if not n.startswith('encoder.')]

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.learning_rate},
            {'params': other_params, 'lr': self.learning_rate}
        ])

        # Calculate total steps and warmup steps based on trainer settings
        max_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max_steps / self.trainer.max_epochs
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    cli = LightningCLI(
        FreqPredModel,
        SicklePretrainDataModule,
        run=True,
    )
    # tuner = Tuner(cli.trainer)
    # lr_finder = tuner.lr_find(cli.model,datamodule=cli.datamodule)
    # print(f"Suggest learning rate: {lr_finder.suggestion()}")
    # cli.model.learning_rate = lr_finder.suggestion()
    # del tuner
    # gc.collect()
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    main()