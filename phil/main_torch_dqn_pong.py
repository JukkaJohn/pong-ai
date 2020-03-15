import gym

from agent.advanced_automatic_agent import AdvancedAutomaticAgent
from environment.pong import Pong, PLAYER_WIDTH, Positions
from nn.net import get_model_input
from phil.simple_dqn_torch import DeepQNetwork, Agent
from utils.plot_learning import plotLearning
import numpy as np
from gym import wrappers

if __name__ == '__main__':
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3,
                  input_dims=6, alpha=0.003)
    opponent = AdvancedAutomaticAgent(PLAYER_WIDTH, player=2, exploration_rate=0.4)

    scores = []
    losses = []
    eps_history = []
    num_games = 500
    score = 0
    loss = 0
    end_score = 3
    # uncomment the line below to record every episode.
    # env = wrappers.Monitor(env, "tmp/space-invaders-1",
    # video_callable=lambda episode_id: True, force=True)
    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
            avg_loss = np.mean(losses[max(0, i - 10):(i + 1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  ' average loss %.3f' % avg_loss,
                  'epsilon %.3f' % brain.EPSILON)
        else:
            print(f'episode: {i}, score: {score}, loss: {loss}')
        pong = Pong(brain, end_score=end_score)
        eps_history.append(brain.EPSILON)
        done = False
        positions = Positions(pong.ball.x, pong.ball.y, pong.player_bottom.rect.x, pong.player_top.rect.x)
        observation = get_model_input(pong.ball.x, pong.ball.y, pong.ball.x, pong.ball.y, pong.player_top.rect.x,
                                      pong.player_bottom.rect.x)
        score = 0
        loss = 0
        while not done:
            opponent_action = opponent.get_direction(None, pong.ball.x, pong.ball.y, pong.player_bottom.rect.x,
                                                     pong.player_top.rect.x)
            brain_action = brain.chooseAction(observation)
            done, positions_, reward = pong.step(opponent_action, brain_action)
            score += reward
            observation_ = get_model_input(positions_.ball_x, positions_.ball_y, positions.ball_x,
                                           positions.ball_y, positions_.own_player_x, positions_.opponent_x)
            brain.storeTransition(observation, brain_action, reward, observation_, done)
            observation = observation_
            positions = positions_
            loss += brain.learn()

        scores.append(score)
        losses.append(loss)

    x = [i + 1 for i in range(num_games)]
    filename = str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + \
               str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) + '.png'
    plotLearning(x, scores, eps_history, filename)
