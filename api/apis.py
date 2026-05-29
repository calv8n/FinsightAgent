"""
agent/groq_client.py — Raw HTTP Groq API calls. Zero SDK dependencies.
"""

import json
import os
import time
from dotenv import load_dotenv
from typing import Optional

import requests

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MODEL")


def llm_request(
    system_prompt: str,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    retries: int = 5,
    retry_delay: float = 8.0,  # 429s need longer waits than other errors
) -> Optional[str]:
    """
    POST to Groq /v1/chat/completions with exponential backoff on 429.

    On 429 the Groq response includes a 'retry-after' header (seconds).
    We honour it when present, otherwise fall back to exponential backoff.
    """
    api_key = os.getenv("GROQ_API_KEY").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Get a free key at https://console.groq.com/keys\n"
            "Then: export GROQ_API_KEY=gsk_..."
        )

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

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0

            if status == 429 and attempt < retries:
                # Honour Groq's retry-after header if present
                retry_after = exc.response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else retry_delay * attempt
                print(
                    f"  [groq] 429 rate limit — waiting {wait:.0f}s "
                    f"(attempt {attempt}/{retries})"
                )
                time.sleep(wait)
                continue

            if status in (500, 502, 503, 504) and attempt < retries:
                wait = retry_delay * attempt
                print(f"  [groq] HTTP {status}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue

            print(f"  [groq] HTTP error {status}: {exc}")
            return None

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt < retries:
                wait = retry_delay * attempt
                print(f"  [groq] Network error, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"  [groq] Network error after {retries} attempts: {exc}")
            return None

        except (KeyError, json.JSONDecodeError) as exc:
            print(f"  [groq] Malformed response: {exc}")
            return None

    print(f"  [groq] All {retries} attempts exhausted.")
    return None
