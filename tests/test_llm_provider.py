from types import SimpleNamespace

from ollama import Message

from palace.llm.factory import get_llm_provider
from palace.llm.provider import BaseLLMProvider, LLMResponse, OllamaProvider
from palace.skills.router import MemoryAgent


def test_get_llm_provider_returns_expected_backend():
    assert isinstance(get_llm_provider("ollama", "llama3.1"), BaseLLMProvider)
    assert isinstance(get_llm_provider("gemini", "gemini-3.6-flash"), BaseLLMProvider)


def test_memory_agent_accepts_provider_instance():
    provider = get_llm_provider("ollama", "llama3.1")
    agent = MemoryAgent(model="llama3.1", provider=provider)
    assert agent.provider is provider


def test_llm_response_is_normalized():
    response = LLMResponse(content="hello", tool_calls=[])
    assert response.content == "hello"
    assert response.tool_calls == []


def test_llm_response_coerces_tool_calls_to_ollama_schema():
    assert LLMResponse._coerce_tool_call(
        {"name": "remember", "arguments": {"value": "blue"}}
    ) == {
        "type": "function",
        "function": {
            "name": "remember",
            "arguments": {"value": "blue"},
        },
    }


def test_ollama_provider_sanitizes_assistant_tool_call_history(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=[]))

    monkeypatch.setattr("palace.llm.provider.ollama.chat", fake_chat)
    messages = [
        {"role": "user", "content": "Remember blue"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "remember", "arguments": {"value": "blue"}}
            ],
        },
        {"role": "tool", "tool_name": "remember", "content": "stored"},
    ]

    response = OllamaProvider("llama3.1").chat(messages)

    assert response.content == "Done"
    assistant_message = captured["messages"][1]
    assert assistant_message["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "arguments": {"value": "blue"},
            },
        }
    ]
    Message.model_validate(assistant_message)
