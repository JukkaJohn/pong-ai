JITTER_MARGIN = 15


class AutomaticAgent:
    def __init__(self, player_width):
        self.player_width = player_width

    def get_direction(self, key_state, ball_x, ball_y, own_player_x, opponent_x) -> int:
        if ball_x - self.get_middle_of_paddle(own_player_x) <= JITTER_MARGIN >= self.get_middle_of_paddle(
                own_player_x) - ball_x:
            return 0
        if ball_x < own_player_x + self.player_width / 2:
            return -1
        elif ball_x > own_player_x + self.player_width / 2:
            return 1
        else:
            return 0

    def get_middle_of_paddle(self, own_player_x) -> int:
        return own_player_x + self.player_width / 2
