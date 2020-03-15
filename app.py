import click
import torch

from agent.advanced_automatic_agent import AdvancedAutomaticAgent
from agent.automatic_agent import AutomaticAgent
from agent.human_agent import HumanAgent
from agent.ai_agent import AiAgent
from environment.pong import Pong, PLAYER_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT
from nn.net import Net1Hidden
from training.model_trainer import ModelTrainer

TWO_PLAYER = 'TWO_PLAYER'
AUTOMATIC_PLAYER = 'AUTOMATIC_PLAYER'
ADVANCED_AUTOMATIC_PLAYER = 'ADVANCED_AUTOMATIC_PLAYER'
AI_PLAYER = 'AI_PLAYER'


@click.group()
def cli():
    pass


@cli.command()
@click.option('--game-type',
              type=click.Choice([TWO_PLAYER, AUTOMATIC_PLAYER, ADVANCED_AUTOMATIC_PLAYER, AI_PLAYER],
                                case_sensitive=False),
              default='TWO_PLAYER')
def play(game_type):
    agent = create_agent(game_type)
    pong = Pong(agent)
    pong.play()


@cli.command()
@click.option('--episodes', type=click.INT, default=100)
@click.option('--learning-rate', type=click.FLOAT, default=0.003)
@click.option('--model-save-dir', type=click.Path(exists=True), required=True)
def train(episodes, learning_rate, model_save_dir):
    model_trainer = ModelTrainer(episodes, learning_rate, model_save_dir)
    model_trainer.train_model()


def create_agent(game_type):
    if game_type == AUTOMATIC_PLAYER:
        return AutomaticAgent(PLAYER_WIDTH)
    elif game_type == ADVANCED_AUTOMATIC_PLAYER:
        return AdvancedAutomaticAgent(PLAYER_WIDTH)
    elif game_type == AI_PLAYER:
        model = Net1Hidden()
        model.load_state_dict(torch.load('models/pong.pth'))
        model.eval()
        return AiAgent(SCREEN_WIDTH, SCREEN_HEIGHT, policy_network=model, epsilon_threshold=0.0)
    return HumanAgent()


if __name__ == '__main__':
    cli()
