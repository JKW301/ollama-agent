import tools


def test_dispatch_retries_transient_errors():
    calls = {"n": 0}

    def flaky_tool() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection timeout")
        return "ok"

    original = tools.FUNCTIONS.get("test_flaky")
    tools.FUNCTIONS["test_flaky"] = flaky_tool
    try:
        out = tools.dispatch("test_flaky", {})
        assert out == "ok"
        assert calls["n"] >= 2
    finally:
        if original is None:
            del tools.FUNCTIONS["test_flaky"]
        else:
            tools.FUNCTIONS["test_flaky"] = original
