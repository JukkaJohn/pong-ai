from collections import namedtuple

import pygame
from ball.ball import Ball
from player.player import Player

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PLAYER_WIDTH = 75
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

LEFT = 0
STAY = 1
RIGHT = 2

Positions = namedtuple('Positions', ('ball_x', 'ball_y', 'opponent_x', 'own_player_x'))


class Pong:
    def __init__(self, agent, end_score=10):
        self.agent = agent
        self.score1 = 0
        self.score2 = 0
        self.end_score = end_score

        pygame.init()

        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

        pygame.display.set_caption('Pong')

        pygame.mouse.set_visible(1)

        self.font = pygame.font.Font(None, 36)

        self.background = pygame.Surface(self.screen.get_size())

        self.ball = Ball(WHITE)
        self.balls = pygame.sprite.Group()
        self.balls.add(self.ball)

        self.player_bottom = Player(580, PLAYER_WIDTH, WHITE)
        self.player_top = Player(25, PLAYER_WIDTH, WHITE)

        self.movingsprites = pygame.sprite.Group()
        self.movingsprites.add(self.player_bottom)
        self.movingsprites.add(self.player_top)
        self.movingsprites.add(self.ball)
        self.clock = pygame.time.Clock()

    def play(self):

        done = False
        exit_program = False

        while not exit_program:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_program = True

            if not done:
                key_state = pygame.key.get_pressed()
                action_player_2 = self.agent.get_direction(key_state, self.ball.x, self.ball.y, self.player_top.rect.x,
                                                           self.player_bottom.rect.x)
                done, _, _ = self.step(key_state[pygame.K_RIGHT] - key_state[pygame.K_LEFT] + 1, action_player_2)

            if done:
                text = self.font.render("Game Over", 1, (200, 200, 200))
                textpos = text.get_rect(centerx=self.background.get_width() / 2)
                textpos.top = 50
                self.screen.blit(text, textpos)
                exit_program = True

        pygame.quit()

    def step(self, action_player_bottom, action_player_top) -> (bool, tuple, float):
        self.screen.fill(BLACK)

        self.player_bottom.update(action_player_bottom - 1)
        self.player_top.update(action_player_top - 1)
        self.ball.update()

        reward = 0
        if self.ball.y < 0:
            reward = -10
            self.score1 += 1
            self.ball.reset()
        elif self.ball.y > 600:
            reward = 10
            self.score2 += 1
            self.ball.reset()

        if pygame.sprite.spritecollide(self.player_bottom, self.balls, False):
            reward = -0.5
            diff = (self.player_bottom.rect.x + self.player_bottom.width / 2) - (self.ball.rect.x + self.ball.width / 2)

            self.ball.y = 570
            self.ball.bounce(diff)

        if pygame.sprite.spritecollide(self.player_top, self.balls, False):
            reward = 1
            diff = (self.player_top.rect.x + self.player_top.width / 2) - (self.ball.rect.x + self.ball.width / 2)

            self.ball.y = 40
            self.ball.bounce(diff)

        scoreprint = "Player 1: " + str(self.score1)
        text = self.font.render(scoreprint, 1, WHITE)
        textpos = (0, 0)
        self.screen.blit(text, textpos)

        scoreprint = "Player 2: " + str(self.score2)
        text = self.font.render(scoreprint, 1, WHITE)
        textpos = (300, 0)
        self.screen.blit(text, textpos)

        self.movingsprites.draw(self.screen)

        pygame.display.flip()

        self.clock.tick(30)

        done = False
        if self.score1 == self.end_score or self.score2 == self.end_score:
            done = True
            pygame.quit()
        return done, Positions(self.ball.x, self.ball.y, self.player_bottom.rect.x, self.player_top.rect.x), reward
