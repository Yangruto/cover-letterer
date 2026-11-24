import logging
import os

import asgi
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from workers import WorkerEntrypoint

import logging_config
from agent import run_chat
from models import ChatRequest

logger = logging.getLogger(__name__)

ENV_KEYS = [
    "CLOUDFLARE_TOKEN",
    "Qdrant_API_KEY",
    "GOOGLE_API_KEY",
]

_initialised = False


def _init(env):
    global _initialised
    if _initialised:
        return
    for key in ENV_KEYS:
        value = getattr(env, key, None)
        if value is not None:
            os.environ[key] = str(value)
    _initialised = True


app = FastAPI(title="Cover Letterer AI Agent", description="Frank's AI Cover Letterer")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    return StreamingResponse(
        run_chat(body.message, body.history),
        media_type="text/plain; charset=utf-8",
    )


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        _init(self.env)
        return await asgi.fetch(app, request.js_object, self.env)
