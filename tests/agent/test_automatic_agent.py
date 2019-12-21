from agent.automatic_agent import AutomaticAgent


def test_get_direction():
    agent = AutomaticAgent(80)
    result = agent.get_direction(None, 300, 300, 400)
    assert result == -1

    result = agent.get_direction(None, 300, 300, 300)
    assert result == -1

    result = agent.get_direction(None, 300, 300, 260)
    assert result == 0

    result = agent.get_direction(None, 300, 300, 200)
    assert result == 1
