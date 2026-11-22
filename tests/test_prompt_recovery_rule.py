from core.prompts import build_system_prompt


def test_recovery_rule_present_in_rule_9():
    assert "never repeat the identical call" in " ".join(build_system_prompt().split())
