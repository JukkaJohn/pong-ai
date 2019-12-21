from agent.advanced_automatic_agent import AdvancedAutomaticAgent


def test_get_direction_ball_upward():
    agent = AdvancedAutomaticAgent(80)
    result = agent.get_direction(None, 300, 300, 400)
    assert result == -1

    result = agent.get_direction(None, 350, 250, 385)
    assert result == 1


def test_get_direction_ball_downward():
    agent = AdvancedAutomaticAgent(80)
    result = agent.get_direction(None, 300, 300, 400)
    assert result == -1

    result = agent.get_direction(None, 350, 350, 415)
    assert result == -1

    result = agent.get_direction(None, 400, 400, 300)
    assert result == 1

def test_get_direction_vertical():
    agent = AdvancedAutomaticAgent(80)
    result = agent.get_direction(None, 300, 300, 400)
    assert result == -1

    result = agent.get_direction(None, 300, 250, 415)
    assert result == -1

    result = agent.get_direction(None, 300, 200, 200)
    assert result == 1
