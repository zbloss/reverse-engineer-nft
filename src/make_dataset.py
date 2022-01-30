import json
import os

import click
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.utils import (build_dataframe, download_image, encode_dataframes,
                         extract_metadata, get_meebit_image_url)


@click.command()
@click.option(
    "--path-to-metadata",
    default="data/meebits/metadata.json",
    help="Path to the metadata json file.",
)
@click.option(
    "--path-to-store-data",
    default="data/meebits",
    help="Path to directory to store images in.",
)
def make_dataset(path_to_metadata: str, path_to_store_data: str):
    """
    Extracts key metadata from path_to_metadata file,
    downloads the images into `path_to_store_data/images`
    and stores a CSV containing the other import attributes
    alongside the images.

    Arguments:
        path_to_metadata (str): Filepath to the metadata.json.
        path_to_store_data (str): Path to where you want to store
                                  the images/ directory and
                                  Metadata CSV file.

    Returns:
        None: None

    """

    image_directory = os.path.join(path_to_store_data, "images")
    for path in [path_to_store_data, image_directory]:
        if not os.path.exists(path):
            os.makedirs(path)
    csv_metadata_filepath = os.path.join(path_to_store_data, "metadata.csv")
    with open(path_to_metadata, "r") as f:
        metadata = json.loads(f.read())
        f.close()

    flattened_dataframes = []
    for obj in tqdm(metadata):
        try:
            meebit_metadata = extract_metadata(obj)
            token_id = meebit_metadata["token_id"]

            image_filepath = os.path.join(image_directory, f"{token_id}.png")
            image_url = get_meebit_image_url(token_id)
            status_code = download_image(image_url, image_filepath)
            if status_code == 200:
                flattened_dataframes.append(build_dataframe(meebit_metadata))
        except:
            pass
        
    csv_metadata = encode_dataframes(
        flattened_dataframes,
        columns_to_exclude=[
            "token_id",
            "original_token_address",
            "name",
        ],
    )
    csv_metadata.to_csv(csv_metadata_filepath, index=None)

    return path_to_metadata


if __name__ == "__main__":
    make_dataset()
