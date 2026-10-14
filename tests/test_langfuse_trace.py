from agentic_evals.agents.langfuse_trace import ENV_KEYS, flush_langfuse, langfuse_handler


def test_off_without_keys(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert langfuse_handler() is None
    flush_langfuse()  # a no-op, and must not raise


def test_handler_pins_one_trace_per_sample(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    first, second = langfuse_handler(), langfuse_handler()
    assert first._trace_context["trace_id"] != second._trace_context["trace_id"]
