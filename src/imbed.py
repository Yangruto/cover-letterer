import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

Qdrant_COLLECTION = "coverletter-bge-m3"
Qdrant_URL = "https://19b40d5f-4485-4e11-9607-f2d522b7d18d.australia-southeast1-0.gcp.cloud.qdrant.io"
MODEL = "@cf/baai/bge-m3"
CLOUDFLARE_ACCOUNT_ID = "bf56f3ef9186ce02bb171c5a019d9313"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing secret: {name}")
    return value


async def cf_embed(texts: list[str], account_id: str, token: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=30) as client:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
        )
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"text": texts},
            timeout=60,
        )
        r.raise_for_status()
    return r.json()["result"]["data"]


async def query_qdrant(vector, top_n=3):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{Qdrant_URL.rstrip('/')}/collections/{Qdrant_COLLECTION}/points/query",
            headers={"api-key": _env("Qdrant_API_KEY")},
            json={"query": list(vector), "limit": top_n, "with_payload": True},
        )
        response.raise_for_status()
        points = response.json()["result"]["points"]
    return [[p["payload"]["page_content"], p["score"]] for p in points]


def top_similarity(query, top_n=3):
    vector = asyncio.run(
        cf_embed(query, CLOUDFLARE_ACCOUNT_ID, _env("CLOUDFLARE_TOKEN"))
    )[0]
    response = asyncio.run(query_qdrant(vector, top_n))
    return response


# def check_qdrant_collection(collection_name):
#     if not qdrant_client.collection_exists(collection_name):
#         qdrant_client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
#         )
#         print(
#             f"{collection_name} is not existed, so create a {collection_name} collection."
#         )
#     else:
#         print(f"{collection_name} is existed.")


# def qdrant_upload(texts, vectors, collection_name):
#     qdrant_client.upsert(
#         collection_name,
#         points=[
#             PointStruct(
#                 id=i,
#                 vector=v,
#                 payload={"page_content": t},
#             )
#             for i, v, t in zip([l for l in range(len(texts))], vectors, texts)
#         ],
#     )

if __name__ == "__main__":
    # check_qdrant_collection(Qdrant_COLLECTION)
    # data = pd.read_csv("../docs/resume_chunk.csv")
    # texts = data["content"].to_list()
    # vectors = cf_embed(texts, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_TOKEN)
    # # qdrant_upload(texts, vectors, Qdrant_COLLECTION)
    # docs = top_similarity("我想認識楊孟翰")
    print("test")
