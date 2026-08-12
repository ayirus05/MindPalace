"""Provider abstractions for multiple LLM backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ollama


def _load_environment_file() -> None:
    """Load a project-level .env file into os.environ when present."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        project_root = Path(__file__).resolve().parents[3]
        env_path = project_root / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return

    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


@dataclass
class LLMResponse:
    """Normalized output format shared by all backends."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any | None = None

    @staticmethod
    def _coerce_tool_call(tool_call: Any) -> dict[str, Any]:
        if hasattr(tool_call, "function"):
            function = getattr(tool_call, "function")
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", {}) or {}
            return {"name": name, "arguments": dict(arguments)}

        if isinstance(tool_call, dict):
            if "function" in tool_call:
                function = tool_call.get("function") or {}
                return {
                    "name": function.get("name"),
                    "arguments": dict(function.get("arguments", {}) or {}),
                }
            return {
                "name": tool_call.get("name"),
                "arguments": dict(tool_call.get("arguments", {}) or {}),
            }

        return {
            "name": getattr(tool_call, "name", None),
            "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
        }

    @classmethod
    def from_ollama(cls, response: Any) -> "LLMResponse":
        message = getattr(response, "message", response) or {}
        if isinstance(message, dict):
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
        else:
            content = getattr(message, "content", "") or ""
            tool_calls = getattr(message, "tool_calls", []) or []
        return cls(
            content=str(content or ""),
            tool_calls=[cls._coerce_tool_call(tool) for tool in tool_calls],
            raw=response,
        )


class BaseLLMProvider(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a message list to the provider and return a normalized response."""


class OllamaProvider(BaseLLMProvider):
    """Ollama-backed provider."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if tools:
            payload["tools"] = tools
        response = ollama.chat(**payload)
        return LLMResponse.from_ollama(response)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini-backed provider."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        _load_environment_file()
        api_key = self.kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env or the environment to use the Gemini provider.")

        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self._client = genai
                return self._client
            except ImportError as exc:  # pragma: no cover - dependency failure path
                raise RuntimeError(
                    "Google Gemini SDK is not installed. Run `pip install google-genai` "
                    "or `pip install google-generativeai`."
                ) from exc

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()

        if hasattr(client, "models"):
            from google.genai import types as genai_types

            system_instructions = [
                str(item["content"])
                for item in messages
                if item.get("role") == "system" and item.get("content")
            ]
            prompt_parts: list[str] = []
            for item in messages:
                role = item.get("role")
                content = item.get("content")
                if role == "system":
                    continue
                if role == "tool":
                    prompt_parts.append(
                        f"Tool result ({item.get('tool_name', 'tool')}): {content or ''}"
                    )
                    continue
                if content:
                    prompt_parts.append(f"{role}: {content}")

            config = genai_types.GenerateContentConfig(
                system_instruction="\n\n".join(system_instructions) or None,
                tools=self._convert_tools(tools),
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents="\n\n".join(prompt_parts) or "",
                config=config,
            )
            return LLMResponse(
                content=self._extract_text(response),
                tool_calls=self._extract_tool_calls(response),
                raw=response,
            )

        model = client.GenerativeModel(
            self.model_name,
            system_instruction="\n\n".join(
                item["content"]
                for item in messages
                if item.get("role") == "system" and item.get("content")
            )
            or None,
            tools=self._convert_tools(tools) if tools else None,
        )
        prompt = "\n\n".join(
            f"{item['role']}: {item.get('content', '')}"
            for item in messages
            if item.get("role") != "system" and item.get("content")
        )
        response = model.generate_content(prompt)
        return LLMResponse(
            content=self._extract_text(response),
            tool_calls=self._extract_tool_calls(response),
            raw=response,
        )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> Any | None:
        if not tools:
            return None

        try:
            from google.genai import types as genai_types
        except Exception:
            return tools

        declarations: list[Any] = []
        for tool in tools:
            function_spec = tool.get("function", tool) if isinstance(tool, dict) else tool
            parameters = function_spec.get("parameters") if isinstance(function_spec, dict) else None
            if parameters is None:
                parameters = {"type": "OBJECT", "properties": {}}

            declarations.append(
                genai_types.FunctionDeclaration(
                    name=function_spec.get("name"),
                    description=function_spec.get("description", ""),
                    parameters=parameters,
                )
            )

        return [genai_types.Tool(function_declarations=declarations)]
        
    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            parts = getattr(getattr(candidate, "content", None), "parts", []) or []
            text_parts: list[str] = []
            for part in parts:
                if part is None:
                    continue
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))
            if text_parts:
                return "".join(text_parts)

        # Avoid triggering the SDK warning on mixed text/function-call responses.
        if hasattr(response, "text"):
            try:
                value = response.text
            except Exception:
                value = None
            if value is not None:
                return str(value or "")
        return ""

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        function_calls: list[Any] = []

        if hasattr(response, "function_calls"):
            function_calls = list(response.function_calls or [])
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            function_calls = list(getattr(candidate, "function_calls", []) or [])

        normalized: list[dict[str, Any]] = []
        for call in function_calls:
            if call is None:
                continue
            name = getattr(call, "name", None)
            if not name and isinstance(call, dict):
                name = call.get("name")
            args = getattr(call, "args", {}) or {}
            if not isinstance(args, dict):
                args = dict(args)
            if not args and isinstance(call, dict):
                args = dict(call.get("args", {}) or {})
            normalized.append({"name": name, "arguments": dict(args or {})})

        return normalized


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "LLMResponse",
    "OllamaProvider",
]
