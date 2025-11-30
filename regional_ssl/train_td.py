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
from model import TimeTranslator,Decoder,patchify,unpatchify
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import SicklePretrainDataModule
import math
import gc

SEED = 42
L.seed_everything(SEED)
torch.set_float32_matmul_precision('medium')

class TimeDiffModel(L.LightningModule):
    def __init__(self, model_name, stats_path, learning_rate=1e-4, warmup_epochs=10, img_size=224, patch_size=16, encoder_dim=384, dim=192, depth=3, num_classes=4, dropout_rate=0.2):
        super().__init__()
        # creating encoder
        self.encoder = timm.create_model(model_name,pretrained=False,img_size=img_size,patch_size=patch_size, in_chans=4) # hard-coded channels to 4 for R G B NIR
        del self.encoder.fc_norm
        del self.encoder.head_drop
        del self.encoder.head
        self.encoder.forward = self.encoder.forward_features
        self.encoder.__dict__.pop('forward_head', None)
        
        stats = torch.load(stats_path,map_location=torch.device("cpu"))
        self.norm = Normalize(mean=stats["mean"], std=stats["std"]) # R G B NIR
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        layers = []
        input_dim = encoder_dim * 2  # Concatenated features from both images
        # First layer
        layers.extend([
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers
        for _ in range(self.depth - 2):  # -2 because we already have first and last layers
            layers.extend([
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
        
        # Final layer
        layers.append(nn.Linear(dim, num_classes))  # 4 classes for 0,1,2,3 months
        self.mlp = nn.Sequential(*layers)
        self.img_size = img_size
        self.patch_size = patch_size
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward_encoder(self, images):
        x = self.encoder(images)[:,1:,...] # without cls token
        return x

    def forward_loss(self,pred,target):
        loss = F.cross_entropy(pred,target)
        return loss

    def training_step(self, batch, batch_idx):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        target = m2 - m1 + 12*(y2-y1) # target month
        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        x = torch.cat((x1,x2),dim=0)
        z = self.forward_encoder(x)
        z_cls = z[:,0,:] # picking up the cls token
        z1 = torch.cat((z_cls[:B],z_cls[B:]),dim=1) # concat along dim
        z2 = torch.cat((z_cls[B:],z_cls[:B]),dim=1) # concat along dim
        pred1 = self.mlp(z1)
        pred2 = self.mlp(z2)
        loss1 = self.forward_loss(pred1,target)
        loss2 = self.forward_loss(pred2,target)
        loss = loss1 + loss2
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        target = m2 - m1 + 12*(y2-y1) # target month
        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        x = torch.cat((x1,x2),dim=0)
        z = self.forward_encoder(x)
        z_cls = z[:,0,:] # picking up the cls token
        z1 = torch.cat((z_cls[:B],z_cls[B:]),dim=1) # concat along dim
        z2 = torch.cat((z_cls[B:],z_cls[:B]),dim=1) # concat along dim
        pred1 = self.mlp(z1)
        pred2 = self.mlp(z2)
        loss1 = self.forward_loss(pred1,target)
        loss2 = self.forward_loss(pred2,target)
        loss = loss1 + loss2
        # Log validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self,batch):
        x1,x2,m1,m2,y1,y2 = batch["chip1"],batch["chip2"],batch["month1"],batch["month2"],batch["year1"],batch["year2"]
        target = m2 - m1 + 12*(y2-y1) # target month
        x1,x2 = self.norm.normalize(x1),self.norm.normalize(x2)
        B = x1.size(0)
        x = torch.cat((x1,x2),dim=0)
        z = self.forward_encoder(x)
        z_cls = z[:,0,:] # picking up the cls token
        z1 = torch.cat((z_cls[:B],z_cls[B:]),dim=1) # concat along dim
        #z2 = torch.cat((z_cls[B:],z_cls[:B]),dim=1) # concat along dim
        pred1 = self.mlp(z1)
        #pred2 = self.mlp(z2)
        return pred1.argmax(dim=1)

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
        TimeDiffModel,
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