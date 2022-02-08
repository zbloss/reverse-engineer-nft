import os
import time
import json
import click
import requests
from tqdm import tqdm


@click.command()
@click.option(
    "--path-to-save-metadata",
    help="filepath to save metadata JSON object to.",
)
@click.option(
    "--x-api-key",
    help="X-API-KEY for Moralis.",
)
@click.option(
    "--token-address",
    default="0x7bd29408f11d2bfc23c34f18275bbf23bb716bc7",
    help="NFT token to search.",
)
@click.option("--blockchain", default="eth", help="Blockchain to search over.")
def get_metadata(
    path_to_save_metadata: str, x_api_key: str, token_address: str, blockchain: str
):
    """
    Utilizes the Moralis API to gather metadata about
    the provided `token_address`.

    Arguments:
        path_to_save_metadata (str): Filepath where you want to save the metadata.
        x_api_key (str): X-API-KEY for Moralis API.
        token_address (str): NFT Token address you want to search.
        blockchain (str): Blockchain you want to search over.

    Returns:
        None: None.

    """

    # making initial request
    base_url = f"https://deep-index.moralis.io/api/v2/nft/{token_address}?chain={blockchain}&format=decimal"
    headers = {"accept": "application/json", "X-API-KEY": x_api_key}
    response = requests.get(f"{base_url}&offset=0&limit=500", headers=headers)
    assert response.status_code == 200
    metadata = response.json()
    total = int(metadata["total"])
    page_size = int(metadata["page_size"])
    offset = page_size

    results = metadata["result"]
    for idx in tqdm(
        range(offset, total + offset, page_size), total=total / page_size - 1
    ):
        response = requests.get(f"{base_url}&offset={idx}&limit=500", headers=headers)
        if response.status_code != 200:
            time.sleep(10)
            response = requests.get(
                f"{base_url}&offset={idx}&limit=500", headers=headers
            )
        metadata = response.json()
        results = results + metadata["result"]
        offset += page_size
    print(f"Retrieved {len(results)} results")

    filepath, filename = os.path.split(path_to_save_metadata)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(path_to_save_metadata, "w") as f:
        f.write(json.dumps(results))
        f.close()

    return None


if __name__ == "__main__":
    get_metadata()
