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


def test_get_direction_remove_jitter():
    agent = AutomaticAgent(80)
    result = agent.get_direction(None, 381, 300, 340)
    assert result == 0

    agent = AutomaticAgent(80)
    result = agent.get_direction(None, 385, 300, 340)
    assert result == 0

    agent = AutomaticAgent(80)
    result = agent.get_direction(None, 379, 300, 340)
    assert result == 0
