from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseLLM:
    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        raise NotImplementedError


class BaseVLM:
    def generate_with_images(
        self,
        system: str,
        user: str,
        image_paths: List[str],
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        raise NotImplementedError


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _require_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "The `openai` package is required for strict API mode. Install it with `pip install openai`."
        ) from e
    return OpenAI


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image path does not exist for VLM call: {p}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class StrictOpenAIChatLLM(BaseLLM):
    """
    Strict API-backed LLM.

    No DummyLLM.
    No silent fallback.
    If the API key, model, base URL, or response is invalid, this class raises immediately.
    """

    def __init__(self, cfg: Dict[str, Any]):
        OpenAI = _require_openai_client()
        provider = str(cfg.get("provider", "openrouter")).lower().strip()

        api_key = (
            cfg.get("api_key")
            or _env_first("OPENAI_API_KEY", "OPENROUTER_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "LLM API key is missing. Set OPENAI_API_KEY or OPENROUTER_API_KEY. "
                "Strict mode refuses to use DummyLLM or fallback responses."
            )

        if provider == "openrouter":
            base_url = cfg.get("base_url") or _env_first("OPENAI_BASE_URL", "OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            model = cfg.get("model") or _env_first("OPENAI_MODEL", "OPENROUTER_MODEL") or "openai/gpt-4o-mini"
        elif provider == "openai":
            base_url = cfg.get("base_url") or _env_first("OPENAI_BASE_URL") or None
            model = cfg.get("model") or _env_first("OPENAI_MODEL") or "gpt-4o-mini"
        else:
            raise RuntimeError(
                f"Unsupported LLM provider in strict mode: {provider}. "
                "Use provider: openrouter or provider: openai."
            )

        self.provider = provider
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        if not str(user).strip():
            raise RuntimeError("LLM user prompt is empty.")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": str(system or "")},
                    {"role": "user", "content": str(user)},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed for model={self.model}: {e}") from e

        try:
            content = resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM API response has no message content: {resp}") from e

        if not content or not str(content).strip():
            raise RuntimeError("LLM API returned an empty response. Strict mode stops instead of using fallback.")
        return str(content).strip()


class StrictOpenAIVLM(BaseVLM):
    """
    Strict API-backed VLM using chat.completions with image_url parts.

    No DummyVLM.
    No silent fallback.
    """

    def __init__(self, cfg: Dict[str, Any]):
        OpenAI = _require_openai_client()
        provider = str(cfg.get("provider", "openrouter")).lower().strip()

        api_key = (
            cfg.get("api_key")
            or _env_first("OPENAI_API_KEY", "OPENROUTER_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "VLM API key is missing. Set OPENAI_API_KEY or OPENROUTER_API_KEY. "
                "Strict mode refuses to use DummyVLM or fallback responses."
            )

        if provider == "openrouter":
            base_url = cfg.get("base_url") or _env_first("OPENAI_BASE_URL", "OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            model = cfg.get("model") or _env_first("OPENAI_VLM_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL") or "openai/gpt-4o-mini"
        elif provider == "openai":
            base_url = cfg.get("base_url") or _env_first("OPENAI_BASE_URL") or None
            model = cfg.get("model") or _env_first("OPENAI_VLM_MODEL", "OPENAI_MODEL") or "gpt-4o-mini"
        else:
            raise RuntimeError(
                f"Unsupported VLM provider in strict mode: {provider}. "
                "Use provider: openrouter or provider: openai."
            )

        self.provider = provider
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_with_images(
        self,
        system: str,
        user: str,
        image_paths: List[str],
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        if not str(user).strip():
            raise RuntimeError("VLM user prompt is empty.")
        if not image_paths:
            raise RuntimeError("VLM image_paths is empty.")

        content: List[Dict[str, Any]] = [{"type": "text", "text": str(user)}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path)}})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": str(system or "")},
                    {"role": "user", "content": content},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        except Exception as e:
            raise RuntimeError(f"VLM API call failed for model={self.model}: {e}") from e

        try:
            out = resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"VLM API response has no message content: {resp}") from e

        if not out or not str(out).strip():
            raise RuntimeError("VLM API returned an empty response. Strict mode stops instead of using fallback.")
        return str(out).strip()


def build_llm(cfg: Dict[str, Any]) -> BaseLLM:
    cfg = cfg or {}
    provider = str(cfg.get("provider", "openrouter")).lower().strip()
    if provider in {"dummy", "mock", "fake", "none", "local_dummy"}:
        raise RuntimeError(
            f"Dummy provider is forbidden in strict final code: {provider}. "
            "Use provider: openrouter or provider: openai."
        )
    return StrictOpenAIChatLLM(cfg)


def build_vlm(cfg: Dict[str, Any]) -> BaseVLM:
    cfg = cfg or {}
    provider = str(cfg.get("provider", "openrouter")).lower().strip()
    if provider in {"dummy", "mock", "fake", "none", "local_dummy"}:
        raise RuntimeError(
            f"Dummy VLM provider is forbidden in strict final code: {provider}. "
            "Use provider: openrouter or provider: openai."
        )
    return StrictOpenAIVLM(cfg)
