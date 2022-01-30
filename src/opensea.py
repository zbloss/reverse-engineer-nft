import time

import requests


class OpenseaAPI:
    def __init__(
        self, asset_owner: str = None, base_api: str = "https://api.opensea.io/api/v1"
    ):

        self.asset_owner = asset_owner
        self.base_api = base_api

    def get_collections(self, maximum_returned_collections: int = 300) -> list:
        """
        Returns all of the collections owned by the provided asset owner.

        Args:
            maximum_returned_collections (int): The maximum number of collections you want to return

        Returns:
            collections (list): All of the collections owned by the asset owner.
        """

        querystring = {
            "asset_owner": self.asset_owner,
            "offset": "0",
            "limit": str(maximum_returned_collections),
        }
        url = f"{self.base_api}/collections"

        response = requests.get(url, params=querystring)
        assert (
            response.status_code == 200
        ), f"Unable to retrieve collections: {response.status_code} | {response.text}"
        return (response.status_code, response.json())

    def _get_assets(
        self, collection: str = None, offset: int = 0, limit: int = 50
    ) -> list:
        """
        Helper method that gathers the maximum number of assets per Opensea API call.

        Args:
            collection (str, optional): Allows you to filter your asset query by collection. Default is None.
            offset (int): How many assets you want to offset in your query. Default is Zero.
            limit (int): Maximum number of assets you want to return per query. Default is Opensea Maximum.

        Returns:
            (status_code, assets) tuple(int, dict): Tuple containing the HTTP response code and dictionary of assets.

        """

        if offset < 0:
            offset = 0

        if limit < 0 or limit > 50:
            limit = 50

        querystring = {
            "owner": self.asset_owner,
            "offset": str(offset),
            "limit": str(limit),
            "order_direction": "desc",
        }
        if collection:
            querystring["collection"] = str(collection)

        url = f"{self.base_api}/assets"

        response = requests.get(url, params=querystring)
        assert (
            response.status_code == 200
        ), f"Unable to retrieve assets: {response.status_code} | {response.text}"
        return (response.status_code, response.json())

    def get_assets(self, collection: str = None):
        """
        Retrieves all assets from an asset owner with the option to filter by collection.

        Args:
            collection (str, optional): Allows you to filter your asset query by collection. Default is None.

        Returns:
            assets list: List of assets as dictionaries.
        """
        params = {
            "offset": 0,
            "limit": 50,
        }

        if collection:
            params["collection"] = str(collection)

        assets = []
        while True:

            time.sleep(2)
            status_code, response = self._get_assets(**params)

            if status_code != 200:
                break

            batch_of_assets = response["assets"]
            assets.extend(batch_of_assets)

            number_of_assets_returned = len(batch_of_assets)
            if number_of_assets_returned < params["limit"]:
                break

            params["offset"] += params["limit"]

        return assets


url = "https://api.opensea.io/api/v1/assets"

for i in range(0, 2):
    querystring = {
        "token_ids": list(range((i * 50) + 1, (i * 50) + 51)),
        "asset_contract_address": "0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7",
        "order_direction": "desc",
        "offset": "0",
        "limit": "50",
    }
    response = requests.request("GET", url, params=querystring)

    print(i, end=" ")
    if response.status_code != 200:
        print("error")
        break

    # Getting meebits data
    meebits = response.json()["assets"]
