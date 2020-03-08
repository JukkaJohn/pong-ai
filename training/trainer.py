import torch
from agent.advanced_automatic_agent import AdvancedAutomaticAgent
from agent.ai_agent import AiAgent
from environment.pong import SCREEN_WIDTH, SCREEN_HEIGHT, Pong, PLAYER_WIDTH
from training.replay_memory import ReplayMemory, Transition

MAX_MEMORY_SIZE = 50000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.996
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(episodes):
    epsilon_threshold = EPSILON_START
    player_1 = AdvancedAutomaticAgent(PLAYER_WIDTH, 580)
    agent = AiAgent(SCREEN_WIDTH, SCREEN_HEIGHT, epsilon_threshold=epsilon_threshold)
    replay_memory = ReplayMemory(MAX_MEMORY_SIZE)

    for i in range(episodes):
        print(f'starting game {i}, epsilon rate = {epsilon_threshold}')
        pong = Pong(agent)

        done = False
        current_state = pong.ball.x, pong.ball.y, pong.player2.rect.x, pong.player1.rect.x
        while not done:
            action_player_1 = player_1.get_direction(None, pong.ball.x, pong.ball.y, pong.player1.rect.x,
                                                     pong.player2.rect.x)
            action_player_2 = agent.get_direction(None, *current_state)
            done, new_state, reward = pong.step(action_player_1, action_player_2)
            replay_memory.push(current_state, action_player_1, new_state, reward)
            current_state = new_state
            epsilon_threshold = epsilon_threshold * EPSILON_DECAY if epsilon_threshold > EPSILON_END else EPSILON_END
            agent.epsilon_threshold = epsilon_threshold


def optimize_model(memory: ReplayMemory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
