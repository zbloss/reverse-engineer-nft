import json
import os
import shutil

import pandas as pd
import requests


def _extract_key(json_object: dict, key_to_extract: str):
    """
    Helper function that attempts to extract `key_to_extract`
    from `json_object`. Implements Try/Except logic.

    Arguments:
        json_object (dict): Dictionary/JSON object.
        key_to_extract (str): Key to attempt to extract
                              `json_object`.

    Returns:
        str: Key value from `json_object` if exists else None.

    """

    extracted_value = None

    try:
        extracted_value = json_object[key_to_extract]
    except KeyError as e:
        print(f"key_to_extract does not exist in json_object: {e}")
    except Exception as e:
        print(f"Unknown Error: {e}")

    return extracted_value


def extract_metadata(meebit_json: dict):
    """
    Given the JSON Response of a Meebit,
    this function extracts the important
    information from it:
        * Token ID
        * Name
        * Attributes
        * Original Token Address

    Arguments:
        meebit_json (dict): JSON Response object containing
                            Meebit Metadata.

    Returns:
        dict: Dictionary containing key fields from `meebit_json`.

    """

    token_id = _extract_key(meebit_json, "token_id")
    metadata = _extract_key(meebit_json, "metadata")
    original_token_address = _extract_key(meebit_json, "token_address")
    metadata = json.loads(metadata)
    name = _extract_key(metadata, "name")
    attributes = _extract_key(metadata, "attributes")

    important_information = {
        "token_id": token_id,
        "original_token_address": original_token_address,
        "name": name,
        "attributes": attributes,
    }

    return important_information


def get_meebit_image_url(meebit_index: int):
    """
    Given a meebit index returns the URL for
    that particular meebit.

    Arguments:
        meebit_index (int): # of the meebit.

    Returns:
        str: URL for that meebit.

    """

    return f"https://meebits.larvalabs.com/meebitimages/characterimage?index={meebit_index}&type=full"


def download_image(url: str, filename: str):
    """
    Given a `url` to an image, this function
    downloads the image to `filename`.

    Arguments:
        url (str): URL to an image.
        filename (str): path to the file save location.

    Returns:
        int: HTTP response code of the HTTP GET request.

    """

    if os.path.isdir(filename):
        filepath, filename = os.path.split(filename)
        os.makedirs(filepath) if not os.path.exists(filepath) else True
        filename = os.path.join(filepath, filename)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        response.raw.decode_content = True
        with open(filename, "wb") as f:
            shutil.copyfileobj(response.raw, f)
            f.close()

    return response.status_code


def build_dataframe(meebit_metadata: dict):
    """
    Given the extracted metadata this will
    format the attributes and metadata into
    a flat dataframe.

    Arguments:
        meebit_metadata (dict): extracted metadata. Typically
                                output of `extract_metadata`.

    Returns:
        pd.DataFrame: Flat DataFrame of important metadata.

    """

    attributes = meebit_metadata.pop("attributes")
    attributes = pd.DataFrame(attributes).T.reset_index(drop=True)
    columns = attributes.loc[0]
    attributes.drop(0, inplace=True)
    attributes.columns = columns
    attributes.reset_index(drop=True, inplace=True)
    other_metadata = pd.DataFrame.from_dict(meebit_metadata, orient="index").T
    flat_dataframe = pd.merge(
        left=attributes,
        right=other_metadata,
        left_index=True,
        right_index=True,
        how="inner",
    )
    return flat_dataframe


def encode_dataframes(list_of_dataframes: list, columns_to_exclude: list):
    """
    Given a list of dataframes with attributes,
    this will one hot encode all of the columns
    except for the `columns_to_exclude`.

    Arguments:
        list_of_dataframes (list): List object containing dataframes.
        columns_to_exclude (list): Columns that should be excluded from
                                   being one-hot encoded from the list of
                                   dataframes.

    Returns:
        pd.DataFrame: One-hot encoded dataframe.

    """

    df = pd.concat(list_of_dataframes)
    excluded_fields_dataframe = df[columns_to_exclude]
    included_fields_dataframe = df.drop(columns_to_exclude, axis=1)

    one_hot_encoded_dataframe = pd.get_dummies(included_fields_dataframe)
    one_hot_encoded_dataframe = pd.merge(
        left=excluded_fields_dataframe,
        right=included_fields_dataframe,
        left_index=True,
        right_index=True,
        how="left",
    )
    one_hot_encoded_dataframe.fillna(0, inplace=True)
    return one_hot_encoded_dataframe