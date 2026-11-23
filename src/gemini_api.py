from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Optional

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

# --- optional imports: depends on installed versions -----------------------
try:
    from google.genai import errors as genai_errors  # newer SDK
except ImportError:  # pragma: no cover
    genai_errors = None

try:
    from google.api_core import exceptions as gexc  # older SDK
except ImportError:  # pragma: no cover
    gexc = None

try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAIError = ()  # type: ignore[assignment]

log = logging.getLogger("gemini_lc")
logging.basicConfig(level=logging.INFO)


class GeminiError(Exception): ...


class GeminiConfigError(GeminiError): ...


class GeminiAuthError(GeminiError): ...


class GeminiBadRequestError(GeminiError): ...


class GeminiQuotaError(GeminiError): ...


class GeminiServerError(GeminiError): ...


class GeminiNetworkError(GeminiError): ...


class GeminiBlockedError(GeminiError): ...


class GeminiEmptyResponseError(GeminiError): ...


class GeminiOutputParseError(GeminiError): ...


RETRYABLE_STATUS = {408, 429, 499, 500, 502, 503, 504}
AUTH_STATUS = {401, 403}

_TEXT_STATUS_HINTS = [  # last resort: sniff the message text
    (429, ("429", "resource_exhausted", "resourceexhausted", "quota", "rate limit")),
    (503, ("503", "unavailable", "overloaded")),
    (504, ("504", "deadline_exceeded", "deadline exceeded", "timed out")),
    (500, ("500", "internal error", "internal server")),
    (403, ("403", "permission_denied", "permission denied")),
    (401, ("401", "unauthenticated", "api key not valid", "api_key_invalid")),
    (404, ("404", "not_found", "not found")),
    (400, ("400", "invalid_argument", "invalid argument")),
]


# ---------------------------------------------------------------------------
# Work out what really went wrong, regardless of library version
# ---------------------------------------------------------------------------
def _exception_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _status_code(exc: BaseException) -> Optional[int]:
    for e in _exception_chain(exc):
        if genai_errors is not None and isinstance(e, genai_errors.APIError):
            return int(e.code)
        if gexc is not None and isinstance(e, gexc.GoogleAPICallError):
            try:
                return int(e.code)
            except (TypeError, ValueError):
                pass
        if isinstance(e, httpx.HTTPStatusError):
            return e.response.status_code
    # fall back to sniffing text of the outermost message
    text = " ".join(str(e).lower() for e in _exception_chain(exc))
    for code, needles in _TEXT_STATUS_HINTS:
        if any(n in text for n in needles):
            return code
    return None


def _is_network_error(exc: BaseException) -> bool:
    return any(
        isinstance(e, (httpx.TransportError, TimeoutError, ConnectionError))
        for e in _exception_chain(exc)
    )


def _retry_hint(exc: BaseException) -> Optional[float]:
    """Parse 'retry in 12.3s' / retryDelay '30s' style hints, if present."""
    for e in _exception_chain(exc):
        if genai_errors is not None and isinstance(e, genai_errors.APIError):
            try:
                for d in (e.details or {}).get("error", {}).get("details", []):
                    if d.get("@type", "").endswith("RetryInfo"):
                        return float(str(d["retryDelay"]).rstrip("s"))
            except Exception:
                pass
    m = re.search(r"retry[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*s", str(exc), re.I)
    return float(m.group(1)) if m else None


def _backoff(attempt: int, base: float, cap: float, hint: Optional[float]) -> float:
    delay = random.uniform(0, min(cap, base * 2**attempt))
    return min(cap, max(delay, hint or 0.0))


# ---------------------------------------------------------------------------
# Check the AIMessage that comes back from .invoke()
# ---------------------------------------------------------------------------
_BLOCKED = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "LANGUAGE",
}


def message_text(msg: Any) -> str:
    """Normalise AIMessage.content (str OR list of blocks) to plain text."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def check_message(msg: Any, *, allow_tool_calls: bool = True) -> Any:
    """Raise if the 'successful' response is blocked, empty, or malformed."""
    if not isinstance(msg, AIMessage):
        return msg  # structured output / parsed objects: nothing to inspect

    meta = getattr(msg, "response_metadata", None) or {}

    feedback = meta.get("prompt_feedback") or {}
    if isinstance(feedback, dict) and feedback.get("block_reason"):
        raise GeminiBlockedError(f"Prompt blocked: {feedback}")

    finish = str(meta.get("finish_reason", "") or "").upper().split(".")[-1]
    if finish in _BLOCKED:
        raise GeminiBlockedError(f"Response blocked, finish_reason={finish}")
    if finish == "MALFORMED_FUNCTION_CALL":
        raise GeminiBadRequestError("Model produced a malformed function call.")

    if allow_tool_calls and getattr(msg, "tool_calls", None):
        return msg  # tool call responses legitimately have empty text

    if not message_text(msg).strip():
        raise GeminiEmptyResponseError(
            f"Empty response (finish_reason={finish or 'n/a'})"
        )

    if finish == "MAX_TOKENS":
        log.warning("Output was truncated (MAX_TOKENS).")
    return msg


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------
def _safe_invoke(
    model: Any,
    messages: Any,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_empty: bool = True,
    retry_parse_errors: int = 1,
    **invoke_kwargs: Any,
) -> Any:
    """
    Drop-in replacement for `model.invoke(messages)` with full error handling.
    Works with plain models and with `model.with_structured_output(...)`.
    """
    last: Optional[BaseException] = None
    parse_failures = 0

    for attempt in range(max_retries + 1):
        try:
            result = model.invoke(messages, **invoke_kwargs)
            return check_message(result)

        # Structured-output / parser failures (model returned bad JSON, etc.)
        except OutputParserException as e:
            last = e
            parse_failures += 1
            if parse_failures > retry_parse_errors:
                raise GeminiOutputParseError(str(e)) from e
            log.warning("Output parse error, retrying: %s", e)

        # Our own raised errors: blocked/bad request are final; empty may retry
        except GeminiEmptyResponseError as e:
            last = e
            if not retry_empty or attempt == max_retries:
                raise
            log.warning("Empty response, retrying.")

        except GeminiError:
            raise

        # Everything else: classify and decide
        except Exception as e:  # noqa: BLE001 - we deliberately classify here
            last = e
            code = _status_code(e)
            log.warning(
                "Attempt %d/%d failed (status=%s): %s",
                attempt + 1,
                max_retries + 1,
                code,
                e,
            )

            if code in AUTH_STATUS:
                raise GeminiAuthError(str(e)) from e

            if code is not None and code not in RETRYABLE_STATUS:
                # 400/404/413/422...: location not supported, bad model name,
                # invalid arguments, input too long, etc.
                raise GeminiBadRequestError(str(e)) from e

            if code is None and not _is_network_error(e):
                # Unknown error type (LangChain ValueError, pydantic error...).
                if isinstance(e, ChatGoogleGenerativeAIError) or isinstance(
                    e, (ValueError, TypeError)
                ):
                    raise GeminiBadRequestError(str(e)) from e
                raise  # truly unexpected: don't hide it

            if attempt == max_retries:
                break
            time.sleep(_backoff(attempt, base_delay, max_delay, _retry_hint(e)))
            continue

        # parse-error branch falls through to here: small pause, then retry
        if attempt < max_retries:
            time.sleep(_backoff(attempt, 0.5, 5.0, None))

    # Retries exhausted
    code = _status_code(last) if last else None
    if code == 429:
        raise GeminiQuotaError(str(last)) from last
    if code is not None:
        raise GeminiServerError(str(last)) from last
    if last is not None and _is_network_error(last):
        raise GeminiNetworkError(str(last)) from last
    raise GeminiEmptyResponseError(str(last)) from last


def safe_invoke(gemini_model, query):
    try:
        resp = _safe_invoke(gemini_model, query)
        return resp
    except GeminiAuthError as e:
        print(f"Auth/permission problem: {e}")
        return f"Auth/permission problem: {e}"
    except GeminiBadRequestError as e:
        print(f"Bad request: {e}")
        return f"Bad request: {e}"
    except GeminiBlockedError as e:
        print(f"Blocked by safety filters: {e}")
        return f"Blocked by safety filters: {e}"
    except GeminiQuotaError as e:
        print(f"Rate limit/quota exceeded: {e}")
        return f"Rate limit/quota exceeded: {e}"
    except GeminiServerError as e:
        print(f"Gemini server trouble: {e}")
        return f"Gemini server trouble: {e}"
    except GeminiNetworkError as e:
        print(f"Network problem: {e}")
        return f"Network problem: {e}"
    except GeminiOutputParseError as e:
        print(f"Could not parse model output: {e}")
        return f"Could not parse model output: {e}"
    except GeminiEmptyResponseError as e:
        print(f"No content returned: {e}")
        return f"No content returned: {e}"
    except Exception:
        log.exception("Unexpected error")
        return "Unexpected error"
