from agent import _normalize_safety_mode, _build_system_prompt, _needs_tools, Agent


def test_normalize_safety_mode():
    assert _normalize_safety_mode("open") == "open"
    assert _normalize_safety_mode(" STRICT ") == "strict"
    assert _normalize_safety_mode("unknown") == "balanced"
    assert _normalize_safety_mode(None) == "balanced"


def test_build_system_prompt_contains_generalist_instruction():
    prompt = _build_system_prompt("open")
    assert "assistant" in prompt.lower()
    assert "auto-disclaimer" in prompt


def test_agent_reset_keeps_system_message():
    a = Agent(safety_mode="open")
    a.messages.append({"role": "user", "content": "bonjour"})
    a.reset()
    assert len(a.messages) == 1
    assert a.messages[0]["role"] == "system"


def test_needs_tools_detects_file_intent_words():
    q = "Redige un essay et sauvegarde le dans un fichier txt"
    assert _needs_tools(q) is True


def test_needs_tools_detects_paths_and_commands():
    assert _needs_tools("ecris ca dans ~/Downloads/note.txt") is True
    assert _needs_tools("git status") is True


def test_needs_tools_ignores_plain_chitchat():
    assert _needs_tools("comment tu vas ?") is False
    assert _needs_tools("que penses-tu de ce livre ?") is False
