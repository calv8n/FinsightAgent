import json
import os
import time
import requests
import logging
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")


def llm_request(
    system_prompt: str,
    messages: list[dict],
    model: str = MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[str]:
    """
    POST to Groq /v1/chat/completions.

    Args:
        system_prompt: Injected as {"role": "system"} at index 0.
        messages:      Windowed history dicts from ConversationHistory.as_dicts().
        retries:       Number of retry attempts on transient errors.

    Returns:
        Assistant response text, or None on unrecoverable error.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            # 429 = rate limit, 5xx = server error → retry
            if status in (429, 500, 502, 503, 504) and attempt < retries:
                wait = retry_delay * attempt
                print(
                    f"  [groq] HTTP {status}, retrying in {wait:.1f}s (attempt {attempt}/{retries})"
                )
                time.sleep(wait)
                last_error = exc
                continue
            print(f"  [groq] HTTP error {status}: {exc}")
            return None

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt < retries:
                wait = retry_delay * attempt
                print(
                    f"  [groq] Network error, retrying in {wait:.1f}s (attempt {attempt}/{retries})"
                )
                time.sleep(wait)
                last_error = exc
                continue
            print(f"  [groq] Network error after {retries} attempts: {exc}")
            return None

        except (KeyError, json.JSONDecodeError) as exc:
            print(f"  [groq] Malformed response: {exc}")
            return None

    print(f"  [groq] All {retries} attempts failed. Last error: {last_error}")
    return None
