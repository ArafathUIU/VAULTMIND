"""Tests for agent behavior."""


def test_agent_modules_import() -> None:
    """Verify agent scaffold modules are importable."""
    import agents.base_agent
    import agents.critic_agent
    import agents.orchestrator
    import agents.reasoning_agent
    import agents.retriever_agent
    import agents.router_agent

    assert agents.router_agent is not None
