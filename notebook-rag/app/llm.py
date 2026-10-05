"""OpenAI Chat Completions client with retries, model fallback and exact token usage."""
import time

import requests

from . import config


class LLMError(RuntimeError):
    """An error whose message is safe and useful to show to the user."""


def _post(model, messages):
    return requests.post(
        f"{config.OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_completion_tokens": config.MAX_OUTPUT_TOKENS},
        timeout=config.LLM_TIMEOUT)


def _error_code(resp):
    try:
        e = resp.json().get("error", {})
        return e.get("code") or e.get("type") or "", e.get("message", "")
    except ValueError:
        return "", resp.text[:200]


def chat(messages: list) -> dict:
    """Send messages, return {"text", "input_tokens", "output_tokens", "model"}."""
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    models = [config.OPENAI_MODEL] + [m for m in config.OPENAI_FALLBACK_MODELS if m != config.OPENAI_MODEL]
    last = ""
    for model in models:
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                r = _post(model, messages)
            except requests.RequestException as exc:
                last = f"network error: {exc}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                j = r.json()
                choice = j["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                if not text:
                    raise LLMError("The model returned an empty answer (output limit reached while thinking). "
                                   "Raise MAX_OUTPUT_TOKENS in .env.")
                u = j.get("usage", {})
                return {"text": text, "input_tokens": u.get("prompt_tokens", 0),
                        "output_tokens": u.get("completion_tokens", 0), "model": j.get("model", model)}
            code, msg = _error_code(r)
            if code == "insufficient_quota":
                raise LLMError("OpenAI says your account has no credit. Add billing credit at platform.openai.com.")
            if r.status_code == 401:
                raise LLMError("OpenAI rejected the API key (401). Check OPENAI_API_KEY.")
            if r.status_code in (404, 400) and code in ("model_not_found", "invalid_model", "unsupported_value",
                                                        "unsupported_parameter"):
                last = f"{model}: {msg}"
                break  # try the next model
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{model}: {r.status_code} {msg}"
                time.sleep(min(2 ** (attempt + 1), 20))
                continue
            raise LLMError(f"OpenAI error {r.status_code}: {msg}")
    raise LLMError(f"All models failed. Last error: {last}")
