import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import logging_config
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from src.agent import build_agent
from src.models import ChatRequest, ChatResponse

# from workers import asgi

logger = logging.getLogger(__name__)

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://jp.cloud.langfuse.com"

langfuse = get_client()
langfuse_handler = CallbackHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = build_agent()
    app.state.agent = agent
    yield


app = FastAPI(
    title="Career Match AI Agent",
    description="Frank's Career Match AI agent",
    lifespan=lifespan,
)
# Default = asgi.entrypoint(app)  # Cloudflare entrypoint


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat/stream", response_model=ChatResponse)
async def chat_stream(body: ChatRequest):
    thread_id = body.thread_id or str(uuid.uuid4())

    def generate():
        logger.debug(f"Thread ID: {thread_id}")
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
        }

        response = app.state.agent.stream(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
            stream_mode="messages",
        )
        for message_chunk, metadata in response:
            if message_chunk.content:
                yield message_chunk.content

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Thread-ID": thread_id,
        },
    )
