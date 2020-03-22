from environment.pong import STAY, LEFT, RIGHT

JITTER_MARGIN = 15


class AutomaticAgent:
    def __init__(self, player_width, player=1):
        self.player_width = player_width

    def get_direction(self, key_state, ball_x, ball_y, own_player_x, opponent_x) -> int:
        if ball_x - self.get_middle_of_paddle(own_player_x) <= JITTER_MARGIN >= self.get_middle_of_paddle(
                own_player_x) - ball_x:
            return STAY
        if ball_x < own_player_x + self.player_width / 2:
            return LEFT
        elif ball_x > own_player_x + self.player_width / 2:
            return RIGHT
        else:
            return STAY

    def get_middle_of_paddle(self, own_player_x) -> int:
        return own_player_x + self.player_width / 2
