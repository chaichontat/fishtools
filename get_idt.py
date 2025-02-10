# %%

import json
import logging
import os
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from urllib import parse, request

import polars as pl
import requests


def get_access_token(client_id: str, client_secret: str, idt_username: str, idt_password: str):
    """
    Create the HTTP request, transmit it, and then parse the response for the
    access token.

    The body_dict will also contain the fields "expires_in" that provides the
    time window the token is valid for (in seconds) and "token_type".
    """

    # Construct the HTTP request
    authorization_string = b64encode(bytes(client_id + ":" + client_secret, "utf-8")).decode()
    request_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic " + authorization_string,
    }

    data_dict = {
        "grant_type": "password",
        "scope": "test",
        "username": idt_username,
        "password": idt_password,
    }
    request_data = parse.urlencode(data_dict).encode()
    print(request_data)
    print(request_headers)

    post_request = request.Request(
        "https://www.idtdna.com/Identityserver/connect/token",
        data=request_data,
        headers=request_headers,
        method="POST",
    )

    # Transmit the HTTP request and get HTTP response
    try:
        response = request.urlopen(post_request)
    except Exception as e:
        print(e)
        raise e

    # Process the HTTP response for the desired data
    body = response.read().decode()

    # Error and return the response from the endpoint if there was a problem
    if response.status != 200:
        raise RuntimeError("Request failed with error code:" + response.status + "\nBody:\n" + body)

    body_dict = json.loads(body)
    return body_dict["access_token"]


client_id = os.environ["client_id"]
client_secret = os.environ["client_secret"]
idt_username = os.environ["idt_username"]
idt_password = os.environ["idt_password"]

token = get_access_token(client_id, client_secret, idt_username, idt_password)


# %%


BASE_URL = "https://www.idtdna.com/restapi"
headers = {"Authorization": "Bearer " + token}
orders = requests.get(BASE_URL + "/v1/SalesOrders", headers=headers).json()

# %%


logging.basicConfig(level=logging.INFO)

futs = []
with ThreadPoolExecutor(max_workers=10) as executor:
    for order in orders["SalesOrders"]:
        futs.append(
            executor.submit(
                requests.get,
                BASE_URL + f"/v1/SalesOrders/{order['SalesOrderNumber']}/LineItems",
                headers=headers,
            )
        )

futs = [fut.result().json() for fut in futs]

# %%


lineitems = list(chain.from_iterable(map(lambda x: x["SalesOrders"], futs)))


# %%


def get(url: str, headers: dict):
    print(url)
    return requests.get(url, headers=headers)


futs_mfg = []
with ThreadPoolExecutor(max_workers=10) as executor:
    for item in lineitems:
        if item["MfgId"] is not None:
            futs_mfg.append(
                executor.submit(
                    get,
                    BASE_URL + f"/v1/SalesOrders/Specs/{item['MfgId']}",
                    headers=headers,
                )
            )

# %%
futs_mfg = [fut.result().json() for fut in futs_mfg]
# %%

# %%
pl.from_dicts(futs_mfg).drop(["_links", "Message"]).filter(pl.col("Sequence").is_not_null()).write_csv(
    "idt.csv"
)
# %%
