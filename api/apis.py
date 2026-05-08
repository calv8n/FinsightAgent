import json
import os
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
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Raw HTTP POST to Groq API.

    Args:
        system_prompt: System instruction
        messages: Message history as {"role": "user"|"assistant", "content": "..."} dicts
        temperature: Sampling temperature (0.0-2.0)

    Returns:
        Assistant response text, or None if error
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Export it: export GROQ_API_KEY=your-key"
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    try:
        response = requests.post(
            GROQ_API_URL, json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Response Parse Error: {e}")
        return None
