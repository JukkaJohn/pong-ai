import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.pong import SCREEN_WIDTH, SCREEN_HEIGHT


class Net1Hidden(nn.Module):

    def __init__(self, input_dims: int = 6, fc_dims: int = 128):
        super(Net1Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, 3)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Net2Hidden(nn.Module):
    def __init__(self, input_dims: int = 6, fc1_dims: int = 256, fc2_dims: int = 256):
        super(Net2Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


def get_model_input(ball_x, ball_y, ball_x_previous, ball_y_previous, own_player_x, opponent_x) -> torch.Tensor:
    return torch.Tensor(
        [ball_x / SCREEN_WIDTH,
         ball_y / SCREEN_HEIGHT,
         ball_x_previous / SCREEN_WIDTH,
         ball_y_previous / SCREEN_HEIGHT,
         own_player_x / SCREEN_WIDTH,
         opponent_x / SCREEN_WIDTH])
