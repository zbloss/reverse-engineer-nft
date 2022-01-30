import os
import json
import click
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
def train_model(path_to_image_dir: str, path_to_metadata: str, max_epochs: int, fast_dev_run: bool, batch_size: int):
    """Executes model training."""

    assert os.path.isdir(path_to_image_dir), f'path_to_image_dir ({path_to_image_dir}) does not exist.'
    assert os.path.isfile(path_to_metadata), f'path_to_metadata ({path_to_metadata}) does not exist.'

    with open(path_to_metadata, 'r') as f:
        metadata = json.loads(f.read())
        f.close()

    dataloader = MeebitDataLoader(path_to_image_dir, metadata, batch_size=batch_size)
    dataloader.setup()

    model = NFTReverseEngineer()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./models',
        monitor='val_loss', 
        save_top_k=1, 
        save_weights_only=True
    )

    mlf_logger = MLFlowLogger(experiment_name="meebits", tracking_uri="file:./ml-runs")
    pl.seed_everything(101, workers=True)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=mlf_logger, 
        deterministic=True,
        accelerator="auto",
        fast_dev_run=fast_dev_run, 
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataloader)
    torch.save()

if __name__ == '__main__':
    train_model()