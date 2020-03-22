import os

import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent.automatic_agent import AutomaticAgent
from environment.pong import Pong, Positions, PLAYER_WIDTH
from nn.net import get_model_input

os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate', type=float, default=1e-2, metavar='N',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


opponent = AutomaticAgent(PLAYER_WIDTH)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        pong = Pong(None, end_score=3)
        positions = Positions(pong.ball.x, pong.ball.y, pong.player_bottom.rect.x, pong.player_top.rect.x)
        observation = get_model_input(pong.ball.x, pong.ball.y, pong.ball.x, pong.ball.y, pong.player_top.rect.x,
                                      pong.player_bottom.rect.x)
        done = False
        ep_reward = 0
        while not done:
            action = select_action(np.array(observation))
            opponent_action = opponent.get_direction(None, pong.ball.x, pong.ball.y, pong.player_bottom.rect.x,
                                                     pong.player_top.rect.x)
            done, positions_, reward = pong.step(opponent_action, action)
            observation_ = get_model_input(positions_.ball_x, positions_.ball_y, positions.ball_x,
                                           positions.ball_y, positions_.own_player_x, positions_.opponent_x)
            policy.rewards.append(reward)
            ep_reward += reward
            observation = observation_
            positions = positions_
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
