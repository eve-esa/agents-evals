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


def test_scores_keep_each_scorer_and_key_separate():
    from inspect_ai.scorer import Score

    from agentic_evals.agents.langfuse_scores import langfuse_scores

    rows = langfuse_scores(
        {
            "geobenchx_scorer": Score(value={"tao": 1.0, "match": True}, explanation="ok"),
            "judge": Score(value="C"),
            "skipped": Score(value={"missing": None}),
        }
    )
    assert [(r["name"], r["value"], r["data_type"]) for r in rows] == [
        ("geobenchx_scorer/tao", 1.0, "NUMERIC"),
        ("geobenchx_scorer/match", 1.0, "BOOLEAN"),
        ("judge", "C", "CATEGORICAL"),
    ]
