import json
import os
import time
from dotenv import load_dotenv
from typing import Optional

import requests

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
]

DEFAULT_MODEL = MODEL_FALLBACK_CHAIN[0]


def llm_request(
    system_prompt: str,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    retries: int = 3,  # retries per model before switching
    retry_delay: float = 6.0,  # base delay for non-429 errors
) -> Optional[str]:
    """
    POST to Groq /v1/chat/completions.

    On 429: immediately try the next model in MODEL_FALLBACK_CHAIN.
            Only sleeps if all models are rate-limited (full chain exhausted).
    On 5xx: exponential backoff on the same model, then move on.
    On success: returns the response string.

    Args:
        model: Starting model. Falls back through MODEL_FALLBACK_CHAIN on 429.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Get a free key: https://console.groq.com/keys\n"
            "Then: export GROQ_API_KEY=gsk_..."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build the chain: start from requested model, append remaining fallbacks
    if model in MODEL_FALLBACK_CHAIN:
        start = MODEL_FALLBACK_CHAIN.index(model)
        chain = MODEL_FALLBACK_CHAIN[start:] + MODEL_FALLBACK_CHAIN[:start]
    else:
        chain = [model] + MODEL_FALLBACK_CHAIN  # unknown model first, then chain

    last_error: Optional[str] = None

    for model_candidate in chain:
        result = _call_with_retry(
            model=model_candidate,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            retries=retries,
            retry_delay=retry_delay,
        )

        if result == "RATE_LIMITED":
            # This model is 429'd — try next in chain immediately
            print(f"  [groq] {model_candidate} rate-limited → trying next model")
            last_error = "429"
            continue

        return result  # None on hard failure, str on success

    # Every model in the chain is rate-limited — wait and try once more
    print(f"  [groq] All models rate-limited. Waiting 30s before final retry...")
    time.sleep(30)

    for model_candidate in chain:
        result = _call_with_retry(
            model=model_candidate,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            retries=1,
            retry_delay=retry_delay,
        )
        if result != "RATE_LIMITED":
            return result

    print(f"  [groq] All {len(chain)} models exhausted. Giving up.")
    return None


def _call_with_retry(
    model: str,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    headers: dict,
    retries: int,
    retry_delay: float,
) -> Optional[str]:
    """
    Attempt one model with retries for transient (5xx/network) errors.

    Returns:
        str          — response text on success
        None         — hard failure (bad request, malformed response)
        "RATE_LIMITED" — 429 received; caller should try next model
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            if attempt > 1 or model != DEFAULT_MODEL:
                print(f"  [groq] ✓ Response from {model}")
            return text

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0

            if status == 429:
                return "RATE_LIMITED"  # signal to caller — try next model

            if status in (500, 502, 503, 504) and attempt < retries:
                wait = retry_delay * attempt
                print(
                    f"  [groq] {model} HTTP {status}, retry {attempt}/{retries} in {wait:.0f}s"
                )
                time.sleep(wait)
                continue

            print(f"  [groq] {model} HTTP {status}: {exc}")
            return None

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt < retries:
                wait = retry_delay * attempt
                print(
                    f"  [groq] {model} network error, retry {attempt}/{retries} in {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            print(f"  [groq] {model} network error: {exc}")
            return None

        except (KeyError, json.JSONDecodeError) as exc:
            print(f"  [groq] {model} malformed response: {exc}")
            return None

    return None
