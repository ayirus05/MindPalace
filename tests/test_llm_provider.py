from palace.llm.factory import get_llm_provider
from palace.llm.provider import BaseLLMProvider, LLMResponse
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
