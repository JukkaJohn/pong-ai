import numpy as np


class AdvancedAutomaticAgent:
    def __init__(self, player_width):
        self.player_width = player_width
        self.ball_x_previous = -1
        self.ball_y_previous = -1

    def get_direction(self, key_state, ball_x, ball_y, own_player_x) -> int:
        result = None
        if self.ball_x_previous == -1:
            result = self.get_direction_based_on_ball_x(ball_x, own_player_x)

        if result is None:
            if ball_y < self.ball_y_previous:
                line_parameters = np.linalg.solve(np.array([[self.ball_x_previous, 1], [ball_x, 1]]), np.array([self.ball_y_previous, ball_y]))
                pred_x_intersection = (25-line_parameters[1])/line_parameters[0]

                if pred_x_intersection > own_player_x + self.player_width / 2:
                    result = 1
                elif pred_x_intersection < own_player_x + self.player_width / 2:
                    result = -1
                else:
                    result = 0
            else:
                if own_player_x + self.player_width / 2 > 400:
                    result = -1
                elif own_player_x + self.player_width / 2 < 400:
                    result = 1
                else:
                    result = 0


        self.ball_x_previous = ball_x
        self.ball_y_previous = ball_y

        return result

    def get_direction_based_on_ball_x(self, ball_x, own_player_x):
        if ball_x < own_player_x + self.player_width / 2:
            return -1
        elif ball_x > own_player_x + self.player_width / 2:
            return 1
        else:
            return 0
