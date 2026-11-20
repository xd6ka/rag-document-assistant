"""Provider resolution: the backend is chosen by env, and stale env must not leak.

These cases exist because of real behaviour: a leftover ``QWEN_BASE_URL`` once
pointed a non-DashScope client at a paid endpoint, which is exactly the kind of
bug that only shows up as "why is it calling the wrong thing" at 1 a.m.
"""

from src.config import PROVIDER_PRESETS, resolve_llm


def test_no_credentials_falls_back_to_local():
    cfg = resolve_llm({})
    assert cfg["provider"] == "ollama"
    assert cfg["is_local"] is True
    assert cfg["base_url"] == "http://localhost:11434/v1"
    assert cfg["api_key"] is None


def test_legacy_qwen_key_selects_dashscope():
    cfg = resolve_llm({"QWEN_API_KEY": "sk-test", "QWEN_BASE_URL": "https://example.com/v1"})
    assert cfg["provider"] == "dashscope"
    assert cfg["api_key"] == "sk-test"
    assert cfg["base_url"] == "https://example.com/v1"


def test_local_provider_ignores_legacy_qwen_base_url():
    """The bug this guards: QWEN_BASE_URL must not hijack a local Ollama client."""
    cfg = resolve_llm(
        {"LLM_PROVIDER": "ollama", "QWEN_API_KEY": "sk-test", "QWEN_BASE_URL": "https://dashscope.example/v1"}
    )
    assert cfg["provider"] == "ollama"
    assert cfg["base_url"] == PROVIDER_PRESETS["ollama"]["base_url"]
    assert cfg["api_key"] is None


def test_explicit_overrides_win_over_preset():
    cfg = resolve_llm(
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "sk-deep",
            "LLM_MODEL": "deepseek-reasoner",
            "LLM_BASE_URL": "https://proxy.example/v1",
        }
    )
    assert cfg["model"] == "deepseek-reasoner"
    assert cfg["base_url"] == "https://proxy.example/v1"
    assert cfg["api_key"] == "sk-deep"
    assert cfg["is_local"] is False


def test_local_timeout_is_longer_than_remote():
    assert resolve_llm({"LLM_PROVIDER": "ollama"})["timeout"] > resolve_llm({"LLM_PROVIDER": "openai"})["timeout"]
