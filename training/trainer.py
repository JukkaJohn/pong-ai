from agent.advanced_automatic_agent import AdvancedAutomaticAgent
from agent.ai_agent import AiAgent
from environment.pong import SCREEN_WIDTH, SCREEN_HEIGHT, Pong, PLAYER_WIDTH
from training.replay_memory import ReplayMemory

MAX_MEMORY_SIZE = 50000


def train_model(episodes):
    exploration_rate = 1.0
    player_1 = AdvancedAutomaticAgent(PLAYER_WIDTH, 580)
    agent = AiAgent(SCREEN_WIDTH, SCREEN_HEIGHT, exploration_rate=exploration_rate)
    replay_memory = ReplayMemory(MAX_MEMORY_SIZE)

    for i in range(episodes):
        print(f'starting game {i}, exploration rate = {exploration_rate}')
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
