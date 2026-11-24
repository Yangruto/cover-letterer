class ChatError(Exception):
    """Base class. `str(e)` is for logs; `e.user_message` is safe to show users."""

    retryable = False
    user_message = "系統發生錯誤，請稍後再試。"


class GeminiError(ChatError):
    user_message = "AI 服務暫時發生問題，請稍後再試。"


class QuotaError(GeminiError):
    """HTTP 429 / RESOURCE_EXHAUSTED: rate limit or daily quota reached."""

    user_message = "目前使用量已達上限，請明天再試。"


class ConfigError(GeminiError):
    """Bad/missing API key, no permission, or unknown model name (401/403/404)."""

    user_message = "AI 服務設定有誤，請暫停使用。"


class BadRequestError(GeminiError):
    """HTTP 400: request rejected (e.g. input too long)."""

    user_message = "無法處理這個問題，請換個方式提問。"


class SafetyBlockError(GeminiError):
    """Prompt or answer blocked by Gemini's safety filters."""

    user_message = "這個問題無法回答，請改問與 Frank 專業背景相關的問題。"


class ServerError(GeminiError):
    """HTTP 5xx / UNAVAILABLE / INTERNAL: usually temporary."""

    retryable = True
    user_message = "AI 服務暫時忙碌，請稍後再試。"


class GeminiTimeoutError(GeminiError):
    retryable = True
    user_message = "AI 回應逾時，請再試一次。"


class GeminiNetworkError(GeminiError):
    retryable = True
    user_message = "網路連線發生問題，請再試一次。"


class EmptyResponseError(GeminiError):
    retryable = True
    user_message = "AI 沒有產生回應，請再試一次。"
