import random

import torch

from environment.pong import STAY, LEFT, RIGHT

JITTER_MARGIN = 15


class AiAgent:
    def __init__(self, screen_width, screen_height, policy_network, epsilon_threshold: float = 0.3):
        self.policy_network = policy_network
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.epsilon_threshold = epsilon_threshold
        self.ball_x_previous = -1
        self.ball_y_previous = -1

    def get_direction(self, key_state, ball_x, ball_y, own_player_x, opponent_x) -> int:
        if self.ball_x_previous == -1 and self.ball_y_previous == -1:
            self.ball_x_previous = ball_x
            self.ball_y_previous = ball_y
            return STAY

        if random.random() > self.epsilon_threshold:
            with torch.no_grad():
                result = self.policy_network(self.get_model_input(ball_x, ball_y, own_player_x, opponent_x)).max(0)[
                    1].item()
        else:
            result = random.choice([LEFT, STAY, RIGHT])
        self.ball_x_previous = ball_x
        self.ball_y_previous = ball_y
        return result

    def get_model_input(self, ball_x, ball_y, own_player_x, opponent_x) -> torch.Tensor:
        velocity_ball_x = ball_x - self.ball_x_previous
        velocity_ball_y = ball_y - self.ball_y_previous
        return torch.Tensor([ball_x / self.screen_width, ball_y / self.screen_height, own_player_x / self.screen_width,
                             opponent_x / self.screen_width, velocity_ball_x / self.screen_width,
                             velocity_ball_y / self.screen_height])
