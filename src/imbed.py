import csv
import json
import logging
import os

import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

Qdrant_COLLECTION = "coverletter-bge-m3"
Qdrant_URL = "https://19b40d5f-4485-4e11-9607-f2d522b7d18d.australia-southeast1-0.gcp.cloud.qdrant.io"
Qdrant_API_KEY = os.getenv("Qdrant_API_KEY")
MODEL = "@cf/baai/bge-m3"
CLOUDFLARE_ACCOUNT_ID = "bf56f3ef9186ce02bb171c5a019d9313"
CLOUDFLARE_TOKEN = os.getenv("CLOUDFLARE_TOKEN")

qdrant_client = QdrantClient(url=Qdrant_URL, api_key=Qdrant_API_KEY)


def cf_embed(texts: list[str], account_id: str, token: str) -> list[list[float]]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"text": texts},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["result"]["data"]


def check_qdrant_collection(collection_name):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(
            f"{collection_name} is not existed, so create a {collection_name} collection."
        )
    else:
        print(f"{collection_name} is existed.")


def qdrant_upload(texts, vectors, collection_name):
    qdrant_client.upsert(
        collection_name,
        points=[
            PointStruct(
                id=i,
                vector=v,
                payload={"page_content": t},
            )
            for i, v, t in zip([l for l in range(len(texts))], vectors, texts)
        ],
    )


def top_similarity(query, top_n=3):
    vector = cf_embed(query, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_TOKEN)[0]
    qdrant_client = QdrantClient(url=Qdrant_URL, api_key=Qdrant_API_KEY)
    response = qdrant_client.query_points(
        collection_name=Qdrant_COLLECTION,
        query=vector,
        limit=top_n,
        with_payload=True,
    )
    return [[p.payload["page_content"], p.score] for p in response.points]


if __name__ == "__main__":
    # check_qdrant_collection(Qdrant_COLLECTION)
    # data = pd.read_csv("../docs/resume_chunk.csv")
    # texts = data["content"].to_list()
    # vectors = cf_embed(texts, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_TOKEN)
    # qdrant_upload(texts, vectors, Qdrant_COLLECTION)
    print("test")
