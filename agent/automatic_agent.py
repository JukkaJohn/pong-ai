class AutomaticAgent:
    def __init__(self, player_width):
        self.player_width = player_width

    def get_direction(self, key_state, ball_x, ball_y, own_player_x) -> int:
        if ball_x < own_player_x + self.player_width / 2:
            return -1
        elif ball_x > own_player_x + self.player_width / 2:
            return 1
        else:
            return 0
