import random

import numpy as np
import torch

from nn.net import Net

JITTER_MARGIN = 15


class AiAgent:
    def __init__(self, screen_width, screen_height, epsilon_threshold: float = 0.3):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.net = Net()
        self.epsilon_threshold = epsilon_threshold
        self.ball_x_previous = -1
        self.ball_y_previous = -1

    def get_direction(self, key_state, ball_x, ball_y, own_player_x, opponent_x) -> int:
        if self.ball_x_previous == -1 and self.ball_y_previous == -1:
            self.ball_x_previous = ball_x
            self.ball_y_previous = ball_y
            return 0
        
        if random.random() > self.epsilon_threshold:
            with torch.no_grad():
                pred = self.net(
                    torch.Tensor(self.get_model_input(ball_x, ball_y, own_player_x, opponent_x))).detach().numpy()
                result = np.argmax(pred) - 1
        else:
            result = random.choice([-1, 0, 1])
        self.ball_x_previous = ball_x
        self.ball_y_previous = ball_y
        return result

    def get_model_input(self, ball_x, ball_y, own_player_x, opponent_x):
        velocity_ball_x = ball_x - self.ball_x_previous
        velocity_ball_y = ball_y - self.ball_y_previous
        return [ball_x / self.screen_width, ball_y / self.screen_height, own_player_x / self.screen_width,
                opponent_x / self.screen_width, velocity_ball_x / self.screen_width,
                velocity_ball_y / self.screen_height]
