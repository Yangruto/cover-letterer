import uuid

import requests

API_URL = "http://localhost:8000"
id = str(uuid.uuid4())

# response = requests.post(
#     f"{API_URL}/api/chat/stream",
#     json={"message": "他會RAG嗎？", "thread_id": id},
#     stream=True,
# )

response = requests.post(
    f"{API_URL}/api/chat/stream",
    json={
        "message": "他有資料分析的經驗嗎？",
        "thread_id": "4805b921-b63c-475e-bfde-95e029da5338",
    },
    stream=True,
)
print("check")
