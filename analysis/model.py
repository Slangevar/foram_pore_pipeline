from click import progressbar
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from . import metrics, adaptive_loss as adaptive_loss_module

# torch.set_float32_matmul_precision("medium")



class SegmentationModel(pl.LightningModule):
    """
    The general Segmentation model using segmentation_models_pytorch.
    """

    def __init__(
        self,
        lr=0.001,
        weight_decay=1e-2,
        num_channels=1,
        num_classes=3,
        architecture="Unet",
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        scheduler="cosine",
        max_epochs=100,
        steps_per_epoch=100,
        loss_function="tversky", # "tversky" or "adaptive"
        tversky_alpha=[0.5, 0.5, 0.3],
    ):
        super().__init__()
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.loss_function_name = loss_function
        self.tversky_alpha = tversky_alpha
        
        # Calculate tversky_beta as 1 - alpha
        if isinstance(tversky_alpha, list):
            self.tversky_beta = [1.0 - a for a in tversky_alpha]
        else:
            self.tversky_beta = 1.0 - tversky_alpha

        if loss_function == "adaptive":
            self.loss_fn = adaptive_loss_module.AdaptiveTverskyCELoss(alpha=self.tversky_alpha, beta=self.tversky_beta)
        elif loss_function == "tversky":
            self.loss_fn = lambda y_hat, y, w: metrics.tversky_loss(y_hat, y, alpha=self.tversky_alpha, beta=self.tversky_beta, weight=w)
        elif loss_function == "mcc_ce":
            self.loss_fn = metrics.mcc_ce_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        # Select Model Builder
        if architecture == "Unet":
            model_builder = smp.Unet
        elif architecture == "UnetPlusPlus":
            model_builder = smp.UnetPlusPlus
        elif architecture == "DeepLabV3":
            model_builder = smp.DeepLabV3
        elif architecture == "DeepLabV3Plus":
            model_builder = smp.DeepLabV3Plus
        elif architecture == "FPN":
            model_builder = smp.FPN
        elif architecture == "PSPNet":
            model_builder = smp.PSPNet
        elif architecture == "Linknet":
            model_builder = smp.Linknet
        elif architecture == "MAnet":
            model_builder = smp.MAnet
        elif architecture == "PAN":
            model_builder = smp.PAN
        elif architecture == "UPerNet":
            model_builder = smp.UPerNet
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Initialize SMP Model
        self.model = model_builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )

        self.softmax = nn.Softmax(dim=1)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )

    def forward(self, x):
        # SMP models return logits
        logits = self.model(x)
        output = self.softmax(logits)
        return output

    def set_learning_rate(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, threshold=0.0001
            )
            monitor = "validation/loss"
        elif self.scheduler == "cosine":
            # Cosine Annealing with Warm Restarts
            # T_0: Number of iterations for the first restart
            # T_mult: A factor increases T_i after a restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=1
            )
            monitor = None # Not needed for cosine
        elif self.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                epochs=self.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
            ) 
            monitor = None
            # OneCycleLR needs interval='step'
            config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
            return config
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
        
        if monitor:
             config["lr_scheduler"]["monitor"] = monitor
             
        return config

    def training_step(self, batch, batch_idx):
        # Split batch
        X, y, w = batch

        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = self.loss_fn(y_hat, y, w)

        # Log sigma values if adaptive
        if self.loss_function_name == "adaptive":
             self.log("train/sigma_t", self.loss_fn.log_sigma_t.exp())
             self.log("train/sigma_ce", self.loss_fn.log_sigma_ce.exp())

        # Log all metrics (loss, dice, iou, classwise)
        self._log_metrics("train", loss, y_hat, y, w)

        return loss

    def _log_metrics(
        self,
        set_name: str,
        loss: torch.Tensor,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> None:

        # Use torch.no_grad() for metric calculation
        with torch.no_grad():
            # Convert to hard predictions once for metrics (interpretable)
            y_hat_hard = (y_hat == y_hat.max(dim=1, keepdim=True)[0]).float()
            # y is already one-hot from loader, ensuring float for calculation
            y_hard = y.float()

            # Log losses
            # Log specific loss component key
            loss_name = f"{self.loss_function_name}_loss"
            self.log(f"{set_name}/{loss_name}", loss, prog_bar=True)

            # Log Weighted Dice and IoU
            dice_val = metrics.dice(y_hat_hard, y_hard, weight=w)
            iou_val = metrics.iou(y_hat_hard, y_hard, weight=w)
            self.log(f"{set_name}/dice", dice_val, prog_bar=True)
            self.log(f"{set_name}/iou", iou_val, prog_bar=True)

            # Calculate classwise metrics
            cls_iou = metrics.classwise_iou(y_hat_hard, y_hard, w)
            cls_dice = metrics.classwise_dice(y_hat_hard, y_hard, w)

            # Log classwise metrics
            for i, class_name in enumerate(metrics.CLASS_NAMES):
                self.log(f"{set_name}/{class_name}/iou", cls_iou[i])
                self.log(f"{set_name}/{class_name}/dice", cls_dice[i])

    def validation_step(self, batch, batch_idx) -> None:

        # Split batch
        X, y, w = batch

        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = self.loss_fn(y_hat, y, w)

        # Log metrics to wandb
        self._log_metrics("validation", loss, y_hat, y, w)
