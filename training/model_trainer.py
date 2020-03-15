import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from agent.advanced_automatic_agent import AdvancedAutomaticAgent
from agent.ai_agent import AiAgent
from environment.pong import SCREEN_WIDTH, SCREEN_HEIGHT, Pong, PLAYER_WIDTH, Positions
from nn.net import Net1Hidden
from training.replay_memory import ReplayMemory, Transition
from utils.running_average import RunningAverage

MAX_MEMORY_SIZE = 50_000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.05
GAMMA = 0.95
TARGET_UPDATE = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_input(ball_x, ball_y, ball_x_previous, ball_y_previous, own_player_x, opponent_x) -> torch.Tensor:
    velocity_ball_x = ball_x - ball_x_previous
    velocity_ball_y = ball_y - ball_y_previous
    return torch.Tensor([ball_x / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, own_player_x / SCREEN_WIDTH,
                         opponent_x / SCREEN_WIDTH, velocity_ball_x / SCREEN_WIDTH,
                         velocity_ball_y / SCREEN_HEIGHT])


class ModelTrainer:
    def __init__(self, episodes: int, learning_rate: int, model_save_dir: str):
        self.episodes = episodes
        self.lr = learning_rate
        self.model_save_dir = model_save_dir
        self.policy_network = Net1Hidden()
        self.target_network = Net1Hidden()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        # self.optimizer = optim.RMSprop(self.policy_network.parameters())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.epsilon_threshold = EPSILON_START
        self.player_1 = AdvancedAutomaticAgent(PLAYER_WIDTH, 580)
        self.agent = AiAgent(SCREEN_WIDTH, SCREEN_HEIGHT, self.policy_network, epsilon_threshold=self.epsilon_threshold)
        self.replay_memory = ReplayMemory(MAX_MEMORY_SIZE)
        self.reward_avg = RunningAverage(window_size=1000)
        self.loss_avg = RunningAverage(window_size=1000)

    def train_model(self):
        print(f'Starting training: gamma = {GAMMA}, lr = {self.lr}')
        epsilon_decay = pow(EPSILON_END / EPSILON_START, 1 / self.episodes)
        for i_episode in range(1, self.episodes + 1):
            print(
                f'starting game {i_episode: 4d}/{self.episodes}, epsilon rate = {self.epsilon_threshold:.2f}, avg. reward = {self.reward_avg.average(): .4f}, avg. loss = {self.loss_avg.average():.4f}')
            pong = Pong(self.agent, end_score=2)

            done = False
            current_positions = Positions(pong.ball.x, pong.ball.y, pong.player_bottom.rect.x, pong.player_top.rect.x)
            current_state = get_model_input(pong.ball.x, pong.ball.y, pong.ball.x, pong.ball.y, pong.player_top.rect.x,
                                            pong.player_bottom.rect.x)
            while not done:
                opponent_action = self.player_1.get_direction(None, pong.ball.x, pong.ball.y, pong.player_bottom.rect.x,
                                                              pong.player_top.rect.x)
                agent_action = self.agent.get_direction(None, *current_positions)
                done, new_positions, reward = pong.step(opponent_action, agent_action)
                agent_action = torch.tensor([agent_action], device=device, dtype=torch.long)
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                self.reward_avg.push(reward.item())
                new_state = get_model_input(new_positions.ball_x, new_positions.ball_y, current_positions.ball_x,
                                            current_positions.ball_y, new_positions.own_player_x,
                                            new_positions.opponent_x)
                self.replay_memory.push(current_state, agent_action, new_state, reward)
                current_state = new_state
                current_positions = new_positions
                self.loss_avg.push(self.optimize_model())

            self.epsilon_threshold = self.epsilon_threshold * epsilon_decay if self.epsilon_threshold > EPSILON_END else EPSILON_END
            self.agent.epsilon_threshold = self.epsilon_threshold

            if i_episode % TARGET_UPDATE == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        torch.save(self.policy_network.state_dict(),
                   os.path.join(self.model_save_dir, f'pong_{self.episodes:04d}.pth'))

    def optimize_model(self) -> float:
        if len(self.replay_memory) < BATCH_SIZE:
            return 0
        transitions = self.replay_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        return loss.item()
