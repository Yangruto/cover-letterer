import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

import httpx

from gemini_error_handling import (
    BadRequestError,
    ChatError,
    ConfigError,
    EmptyResponseError,
    GeminiError,
    GeminiNetworkError,
    GeminiTimeoutError,
    QuotaError,
    SafetyBlockError,
    ServerError,
)
from imbed import top_similarity

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
CHAT_MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.7
TOP_N = 3
MAXIMUM_STEP = 5
HTTP_TIMEOUT = 60
SUMMARY_SEPARATOR = "\x1e"
LINKEDIN_FOOTER = "\n\nLinkedIn: https://www.linkedin.com/in/meng-han-yang"

ANSWER_SYSTEM_PROMPT = """
你是 楊孟翰 (Frank) 的推薦人，你想協助他獲得HR的面試邀請。
- 若檢索到的資訊無法回答HR的問題，請直接說你不知道，無相關資訊可回答你的問題。
- 若HR的問題與了 楊孟翰 (Frank) 的特質、背景、技能無關，或是涉及個人隱私，提醒他切勿詢問與求職者專業技能、背景不相關的資訊。
- 任何與了解 楊孟翰 (Frank) 專業技能、人格特質、背景無關的問題，即使簡單無害也不要回覆，並提醒他只能詢問求職者專業技能、背景。
- 楊孟翰是 Frank 的中文名，回覆時擇一即可。
"""

SUMMARY_SYSTEM_PROMPT = """
你是 楊孟翰 (Frank) 的推薦人，幫他寫一篇 cover letter，協助他獲得HR的面試邀請。
- 直接以『我覺得 Frank 是一個非常合適的人選...』開頭。
- 直接從 cover letter 的內容開始。
- 綜合先前HR的問題，著重在技能與人格特質可以如何幫助公司與團隊。

禁止輸出：
- 開場白、問候語或客套話
- 確認使用者要求的語句
- 介紹即將生成內容的文字
- 解釋生成目的、背景或過程的文字
- 任何非使用者要求的補充說明或結尾
"""


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing secret: {name}")
    return value


def _gemini_headers() -> dict:
    return {
        "x-goog-api-key": _env("GOOGLE_API_KEY"),
        "Content-Type": "application/json",
    }


def retrieve(text: str) -> list[float]:
    retrieved_info = top_similarity(text, TOP_N)
    logger.info(f"Retrieved:\n {[i[0] for i in retrieved_info]}")
    retrieved_content = "\n".join([i[0] for i in retrieved_info])
    return retrieved_content


def _classify(status: int, body_text: str) -> GeminiError:
    """Map a non-200 Gemini response to the matching exception."""
    status_str, message = "", body_text[:300]
    try:
        err = json.loads(body_text).get("error", {})
        status_str = err.get("status", "")
        message = err.get("message", message)
    except (ValueError, AttributeError):
        pass
    detail = f"HTTP {status} {status_str}: {message}"

    if status == 429 or status_str == "RESOURCE_EXHAUSTED":
        return QuotaError(detail)
    if status in (401, 403, 404) or status_str in (
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
    ):
        return ConfigError(detail)
    if (
        status == 400 and "api key" in message.lower()
    ):  # Gemini reports a bad key as 400
        return ConfigError(detail)
    if status == 400:
        return BadRequestError(detail)
    if status >= 500 or status_str in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
        return ServerError(detail)
    return GeminiError(detail)


MAX_ATTEMPTS = 2
RETRY_DELAY = 1.0  # seconds, multiplied by the attempt number
BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
}


async def stream_gemini(system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE},
    }
    url = f"{GEMINI_BASE}/models/{CHAT_MODEL}:streamGenerateContent?alt=sse"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        produced = False
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                async with client.stream(
                    "POST", url, headers=_gemini_headers(), json=body
                ) as r:
                    if r.status_code != 200:
                        raw = (await r.aread()).decode("utf-8", "replace")
                        raise _classify(r.status_code, raw)

                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            data = json.loads(payload)
                        except ValueError:
                            logger.warning(
                                "Skipping malformed SSE chunk: %.200s", payload
                            )
                            continue

                        block = (data.get("promptFeedback") or {}).get("blockReason")
                        if block:
                            raise SafetyBlockError(f"prompt blocked: {block}")

                        for cand in data.get("candidates", []):
                            for part in (cand.get("content") or {}).get("parts", []):
                                text = part.get("text")
                                if text:
                                    produced = True
                                    yield text
                            finish = cand.get("finishReason")
                            if finish in BLOCKED_FINISH_REASONS and not produced:
                                raise SafetyBlockError(f"answer blocked: {finish}")

            if not produced:
                raise EmptyResponseError("stream ended without any text")
            return  # success

        except GeminiError as e:
            err: GeminiError = e
        except httpx.TimeoutException as e:  # must come before TransportError
            err = GeminiTimeoutError(f"{type(e).__name__}: {e}")
        except httpx.TransportError as e:
            err = GeminiNetworkError(f"{type(e).__name__}: {e}")

        logger.warning(
            "Gemini attempt %d/%d failed: %s: %s",
            attempt,
            MAX_ATTEMPTS,
            type(err).__name__,
            err,
        )
        if produced or not err.retryable or attempt == MAX_ATTEMPTS:
            raise err
        await asyncio.sleep(RETRY_DELAY * attempt)


def _answer_prompt(question: str, retrieved: str) -> str:
    return f"""
    **HR的問題**
    {question}

    **檢索到的資訊**
    {retrieved}

    請依據檢索到的資訊回答問題。

    **範例**
    回答HR的問題，用一句話描述 Frank 有什麼樣的人格特質、經歷或是專業技能。

    【一項專業技能或特質】:在哪間公司的什麼經驗，可以帶給下一間公司的好處
    【一項專業技能或特質】:在哪間公司的什麼經驗，可以帶給下一間公司的好處
    列出1至2項。
    """


def _summary_prompt(previous_content: str) -> str:
    return f"""
    先前的問答
    {previous_content}

    依據先前的問答，幫 楊孟翰 (Frank) 寫一篇 cover letter。字數在200到300字。

    **範例**
    我覺得 Frank 是一個非常合適的人選，為什麼？...
    """


# entrypoint
async def run_chat(message: str, history: list) -> AsyncIterator[str]:
    step = sum(1 for t in history if t.role == "user") + 1
    logger.info("Current step: STEP %d", step)

    try:
        # RAG
        retrieved = retrieve(message)
        # Answer
        answer = ""
        async for chunk in stream_gemini(
            ANSWER_SYSTEM_PROMPT, _answer_prompt(message, retrieved)
        ):
            answer += chunk
            yield chunk
        # Summary
        if step >= MAXIMUM_STEP:
            yield SUMMARY_SEPARATOR
            lines = [
                f"{'HR' if t.role == 'user' else 'AI'}: {t.content}" for t in history
            ]
            lines += [f"HR: {message}", f"AI: {answer}"]
            async for chunk in stream_gemini(
                SUMMARY_SYSTEM_PROMPT, _summary_prompt("\n".join(lines))
            ):
                yield chunk
            yield LINKEDIN_FOOTER
    except Exception:
        logger.exception("run_chat failed")
        yield "\n\n（系統發生錯誤，請稍後再試。）"
