from agent.automatic_agent import AutomaticAgent
from environment.pong import PLAYER_WIDTH, Pong, Positions
from nn.net import get_model_input
from phil_policy.reinforce_torch import PolicyGradientAgent
from utils.plot_learning import plotLearning

if __name__ == '__main__':
    agent = PolicyGradientAgent(lr=0.001, input_dims=6, GAMMA=0.99,
                                n_actions=3, layer1_size=128, layer2_size=128)
    opponent = AutomaticAgent(PLAYER_WIDTH)
    # agent.load_checkpoint()
    score_history = []
    score = 0
    loss = 0
    num_episodes = 2500
    for i in range(num_episodes):
        print(f'episode: {i}/{num_episodes} score: {score} loss: {loss}')
        done = False
        pong = Pong(None, end_score=3)
        positions = Positions(pong.ball.x, pong.ball.y, pong.player_bottom.rect.x, pong.player_top.rect.x)
        observation = get_model_input(pong.ball.x, pong.ball.y, pong.ball.x, pong.ball.y, pong.player_top.rect.x,
                                      pong.player_bottom.rect.x)
        score = 0
        while not done:
            opponent_action = opponent.get_direction(None, pong.ball.x, pong.ball.y, pong.player_bottom.rect.x,
                                                     pong.player_top.rect.x)
            brain_action = agent.choose_action(observation)
            done, positions_, reward = pong.step(opponent_action, brain_action)
            observation_ = get_model_input(positions_.ball_x, positions_.ball_y, positions.ball_x,
                                           positions.ball_y, positions_.own_player_x, positions_.opponent_x)
            agent.store_rewards(reward)
            observation = observation_
            positions = positions_
            score += reward

        score_history.append(score)
        loss = agent.learn()
        # agent.save_checkpoint()
    filename = 'pong-alpha001-128x128fc-newG.png'
    plotLearning(score_history, filename=filename, window=25)
