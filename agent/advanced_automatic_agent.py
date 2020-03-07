import numpy as np

JITTER_MARGIN = 15


class AdvancedAutomaticAgent:
    def __init__(self, player_width, player=1):
        self.player_width = player_width
        self.player = player
        self.player_y = 25 if player == 1 else 580
        self.ball_x_previous = -1
        self.ball_y_previous = -1

    def get_direction(self, key_state, ball_x, ball_y, own_player_x, opponent_x) -> int:
        result = None

        if self.ball_x_previous == -1 or ball_x == self.ball_x_previous:
            result = self.get_direction_based_on_ball_x(ball_x, own_player_x)

        if result is None:

            if self.ball_coming_towards_player(ball_y, self.ball_y_previous):

                line_parameters = np.linalg.solve(np.array([[self.ball_x_previous, 1], [ball_x, 1]]),
                                                  np.array([self.ball_y_previous, ball_y]))
                pred_x_intersection = (self.player_y - line_parameters[1]) / line_parameters[0]
                while pred_x_intersection < 0 or pred_x_intersection > 800:
                    if pred_x_intersection > 800:
                        pred_x_intersection = 800 - (pred_x_intersection - 800)

                    if pred_x_intersection < 0:
                        pred_x_intersection = -pred_x_intersection

                result = self.get_direction_based_on_ball_x(pred_x_intersection, own_player_x)
            else:
                result = self.get_direction_based_on_ball_x(400, own_player_x)

        self.ball_x_previous = ball_x
        self.ball_y_previous = ball_y

        return result

    def ball_coming_towards_player(self, current_ball_y, previous_ball_y):
        if self.player == 1:
            return current_ball_y < previous_ball_y
        else:
            return current_ball_y > previous_ball_y

    def get_middle_of_paddle(self, own_player_x) -> int:
        return own_player_x + self.player_width / 2

    def get_direction_based_on_ball_x(self, ball_x, own_player_x):
        if ball_x - self.get_middle_of_paddle(own_player_x) <= JITTER_MARGIN >= self.get_middle_of_paddle(
                own_player_x) - ball_x:
            return 0
        if ball_x < own_player_x + self.player_width / 2:
            return -1
        else:
            return 1
