import os
import torch
import pytorch_lightning as pl

from . import loader, model as model_module



def train_model(
    initial_lr=0.0001,
    weight_decay=1e-2,
    batch_size=8,
    epochs=100,
    n_channels=1,
    n_classes=3,
    architecture="Unet",
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    continue_training=False,
    train_folder="data/train",
    val_folder="data/val",
    filename="model",
    num_workers=8,
    scheduler="cosine",
    loss_function="tversky",
    tversky_alpha=[0.5, 0.5, 0.3],
):

    # Device selection
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available(): # accurate check for MPS
        accelerator = "mps"
    
    # device variable for clarity/logging if needed, though accelerator is what Trainer needs
    print(f"Using accelerator: {accelerator}")

    train_loader = loader.get_data_loader(train_folder, batch_size, augment=True, num_workers=num_workers)
    val_loader = loader.get_data_loader(
        val_folder, batch_size, shuffle=False, augment=False, num_workers=num_workers
    )

    model_path = "model/" + filename + ".ckpt"
    resume_checkpoint_path = None

    # Always instantiate the model with the requested configuration
    # This allows changing non-structural parameters if needed, though strictly 
    # for resuming, dimensions/arch must match.
    model = model_module.SegmentationModel(
        lr=initial_lr,
        weight_decay=weight_decay,
        num_channels=n_channels,
        num_classes=n_classes,
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        scheduler=scheduler,
        max_epochs=epochs,
        steps_per_epoch=len(train_loader),
        loss_function=loss_function,
        tversky_alpha=tversky_alpha,
    )

    if continue_training and os.path.isfile(model_path):
        print(f"Resuming training from checkpoint: {model_path}")
        resume_checkpoint_path = model_path
    elif continue_training:
        print(f"Warning: Checkpoint {model_path} not found. Starting fresh.")

    # Loggers
    csv_logger = pl.loggers.CSVLogger("logs", name=filename)
    tb_logger = pl.loggers.TensorBoardLogger("logs", name=filename)

    # Save best model callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model/",
        filename=filename,
        monitor=f"validation/{loss_function}_loss",
        mode="min",
        save_last=True, # Save last wrapper to ensure we can always resume specific state
    )

    # Train model
    # Note: If resuming, 'epochs' must be greater than the epochs already trained in the checkpoint.
    # e.g. if checkpoint has 100 epochs, set epochs=200 to train for 100 more.
    model.train()
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tb_logger],
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint_path)


