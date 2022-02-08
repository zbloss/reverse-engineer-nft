import os
import json
from random import sample
import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from data.meebit_dataloader import MeebitDataLoader
from models.nft_reverse_engineer import NFTReverseEngineer


@click.command()
@click.option(
    "--path-to-image-dir",
    help="Path the the directory containing the nft images.",
)
@click.option(
    "--path-to-metadata",
    help="path to the metadata.json file.",
)
@click.option(
    "--max-epochs",
    default=10,
    help="Max number of epochs to run.",
)
@click.option(
    "--fast-dev-run",
    default=False,
    help="True if you want to simply test the pipeline.",
)
@click.option(
    "--batch-size",
    default=4,
    help="Max number of epochs to run.",
)
@click.option(
    "--auto-scale-batch-size",
    default=None,
    help="If you want to automatically use the largest available batch size..",
)
@click.option(
    "--auto-lr-find",
    default=False,
    help="Automatically find the best learning rate.",
)
@click.option(
    "--track-grad-norm",
    default=2,
    help="Automatically exploding ('inf') or vanishing gradients (l2).",
)
@click.option(
    "--resume-from-checkpoint",
    default=None,
    help="If you want to resume training from a checkpoint provide the path to that checkpoint.",
)
@click.option(
    "--target-loss-fn",
    default="ssim",
    help="Target loss function you want to use. [ssim, psnr, mse].",
)
@click.option(
    "--use-cuda",
    default=True,
    help="True if you want to use GPU training.",
)
def train_model(
    path_to_image_dir: str,
    path_to_metadata: str,
    max_epochs: int,
    fast_dev_run: bool,
    batch_size: int,
    auto_scale_batch_size: bool,
    auto_lr_find: bool,
    track_grad_norm: str,
    resume_from_checkpoint: str,
    target_loss_fn: str,
    use_cuda: bool,
):
    """Executes model training."""
    assert os.path.isdir(
        path_to_image_dir
    ), f"path_to_image_dir ({path_to_image_dir}) does not exist."
    assert os.path.isfile(
        path_to_metadata
    ), f"path_to_metadata ({path_to_metadata}) does not exist."

    with open(path_to_metadata, "r") as f:
        metadata = json.loads(f.read())
        f.close()

    dataloader = MeebitDataLoader(path_to_image_dir, metadata, batch_size=batch_size)
    dataloader.setup()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./models", monitor="val_loss", save_top_k=1
    )

    mlf_logger = MLFlowLogger(experiment_name="meebits", tracking_uri="file:./ml-runs")
    trainer_params = {
        "max_epochs": int(max_epochs),
        "logger": mlf_logger,
        "deterministic": True,
        "accelerator": "auto",
        "fast_dev_run": bool(fast_dev_run),
        "callbacks": [checkpoint_callback],
        "auto_scale_batch_size": None
        if auto_scale_batch_size is "None"
        else auto_scale_batch_size,
        "auto_lr_find": bool(auto_lr_find),
        "precision": 16,
        "resume_from_checkpoint": None
        if resume_from_checkpoint is "None"
        else resume_from_checkpoint
        #'track_grad_norm': track_grad_norm
    }
    model_params = {
        'target_loss': target_loss_fn
    }
    use_cuda = bool(use_cuda)
    if use_cuda and torch.cuda.is_available():
        trainer_params["gpus"] = 1
        model_params['use_cuda'] = use_cuda
        print("Using available GPU")

    model = NFTReverseEngineer(**model_params)
    pl.seed_everything(101, workers=True)
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train_model()
