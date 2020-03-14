import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.pong import SCREEN_WIDTH, SCREEN_HEIGHT


class Net1Hidden(nn.Module):

    def __init__(self):
        super(Net1Hidden, self).__init__()
        self.fc1 = nn.Linear(6, 120)
        self.fc2 = nn.Linear(120, 3)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Net2Hidden(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Net2Hidden, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


def get_model_input(ball_x, ball_y, ball_x_previous, ball_y_previous, own_player_x, opponent_x) -> torch.Tensor:
    velocity_ball_x = ball_x - ball_x_previous
    velocity_ball_y = ball_y - ball_y_previous
    return torch.Tensor([ball_x / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, own_player_x / SCREEN_WIDTH,
                         opponent_x / SCREEN_WIDTH, velocity_ball_x / SCREEN_WIDTH,
                         velocity_ball_y / SCREEN_HEIGHT])
