from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from eo_gym.runtime.core.llm_logger import log_interaction

OPENAI_TIMEOUT_TOTAL = float(os.getenv("OPENAI_TIMEOUT_TOTAL", "360"))
OPENAI_TIMEOUT_READ = float(os.getenv("OPENAI_TIMEOUT_READ", "240"))
OPENAI_TIMEOUT_WRITE = float(os.getenv("OPENAI_TIMEOUT_WRITE", "60"))
OPENAI_TIMEOUT_CONNECT = float(os.getenv("OPENAI_TIMEOUT_CONNECT", "240"))


def build_openai_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        OPENAI_TIMEOUT_TOTAL,
        read=OPENAI_TIMEOUT_READ,
        write=OPENAI_TIMEOUT_WRITE,
        connect=OPENAI_TIMEOUT_CONNECT,
    )


class ResponsesTransport:
    """Shared helper for calling the Responses API with retries."""

    def __init__(self, *, timeout: Optional[httpx.Timeout] = None):
        self.timeout = timeout or build_openai_timeout()
        self.max_api_retries = 6
        self.api_backoff = 2.0

    def send(self, *, request_json: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com") + "/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_api_retries + 1):
            try:
                resp = httpx.post(url, headers=headers, json=request_json, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait_time = self.api_backoff * attempt
                    print(f"[Retry {attempt}/{self.max_api_retries}] {resp.status_code}. Sleeping {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                log_interaction(
                    source="responses_transport",
                    interaction_type="responses_api_call",
                    input_payload=request_json,
                    response=payload,
                    metadata={"url": url, "attempt": attempt},
                )
                return payload
            except httpx.RequestError as exc:
                wait_time = self.api_backoff * attempt
                print(f"[Network Retry {attempt}/{self.max_api_retries}] {exc}. Sleeping {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            except httpx.HTTPStatusError as exc:
                print(f"[Fatal] HTTP error: {exc}")
                try:
                    print("[Error body]", exc.response.json())
                except Exception:
                    print("[Error raw]", exc.response.text)
                raise

        raise RuntimeError("Failed to call OpenAI Responses API after retries.")


def extract_output_text(payload: Dict[str, Any]) -> str:
    text = payload.get("output_text") or ""
    if text:
        return text

    chunks: List[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "".join(chunks)
