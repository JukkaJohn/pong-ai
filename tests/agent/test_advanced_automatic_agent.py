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


def test_get_direction_remove_jitter_ball_downward():
    agent = AdvancedAutomaticAgent(80)
    result = agent.get_direction(None, 381, 300, 440)
    assert result == -1

    result = agent.get_direction(None, 382, 310, 361)
    assert result == 0


def test_get_direction_remove_jitter_ball_upward():
    agent = AdvancedAutomaticAgent(80)
    agent.get_direction(None, 300, 300, 536)

    result = agent.get_direction(None, 350, 250, 536)
    assert result == 0


def test_get_direction_bounce_against_right_wall():
    agent = AdvancedAutomaticAgent(80)
    agent.get_direction(None, 600, 400, 700)

    result = agent.get_direction(None, 650, 350, 700)
    assert result == -1


def test_get_direction_bounce_against_left_wall():
    agent = AdvancedAutomaticAgent(80)
    agent.get_direction(None, 300, 600, 50)

    result = agent.get_direction(None, 250, 550, 50)
    assert result == 1
