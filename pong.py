import click
import pygame

# Define some colors
from ball.ball import Ball
from player.automaticplayer import AutomaticPlayer
from player.humanplayer import HumanPlayer

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

TWO_PLAYER = 'TWO_PLAYER'
AUTOMATIC_PLAYER = 'AUTOMATIC_PLAYER'


@click.command()
@click.option('--game-type',
              type=click.Choice([TWO_PLAYER, AUTOMATIC_PLAYER], case_sensitive=False), default='TWO_PLAYER')
def pong(game_type):
    score1 = 0
    score2 = 0

    # Call this function so the Pygame library can initialize itself
    pygame.init()

    # Create an 800x600 sized screen
    screen = pygame.display.set_mode([800, 600])

    # Set the title of the window
    pygame.display.set_caption('Pong')

    # Enable this to make the mouse disappear when over our window
    pygame.mouse.set_visible(0)

    # This is a font we use to draw text on the screen (size 36)
    font = pygame.font.Font(None, 36)

    # Create a surface we can draw on
    background = pygame.Surface(screen.get_size())

    # Create the ball
    ball = Ball(WHITE)
    # Create a group of 1 ball (used in checking collisions)
    balls = pygame.sprite.Group()
    balls.add(ball)

    # Create the player paddle object
    player1 = HumanPlayer(580, WHITE)
    player2 = create_player(game_type)

    movingsprites = pygame.sprite.Group()
    movingsprites.add(player1)
    movingsprites.add(player2)
    movingsprites.add(ball)

    clock = pygame.time.Clock()
    done = False
    exit_program = False

    while not exit_program:

        # Clear the screen
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True

        # Stop the game if there is an imbalance of 3 points
        if score1 == 10 or score2 == 10:
            done = True

        if not done:
            # Update the player and ball positions
            key_state = pygame.key.get_pressed()
            player1.update(key_state[pygame.K_RIGHT] - key_state[pygame.K_LEFT], ball.x, ball.y)
            player2.update(key_state[pygame.K_x] - key_state[pygame.K_z], ball.x, ball.y)
            ball.update()

        # If we are done, print game over
        if done:
            text = font.render("Game Over", 1, (200, 200, 200))
            textpos = text.get_rect(centerx=background.get_width() / 2)
            textpos.top = 50
            screen.blit(text, textpos)

        if ball.y < 0:
            score1 += 1
            ball.reset()
        elif ball.y > 600:
            score2 += 1
            ball.reset()

        # See if the ball hits the player paddle
        if pygame.sprite.spritecollide(player1, balls, False):
            # The 'diff' lets you try to bounce the ball left or right depending where on the paddle you hit it
            diff = (player1.rect.x + player1.width / 2) - (ball.rect.x + ball.width / 2)

            # Set the ball's y position in case we hit the ball on the edge of the paddle
            ball.y = 570
            ball.bounce(diff)

        # See if the ball hits the player paddle
        if pygame.sprite.spritecollide(player2, balls, False):
            # The 'diff' lets you try to bounce the ball left or right depending where on the paddle you hit it
            diff = (player2.rect.x + player2.width / 2) - (ball.rect.x + ball.width / 2)

            # Set the ball's y position in case we hit the ball on the edge of the paddle
            ball.y = 40
            ball.bounce(diff)

        # Print the score
        scoreprint = "Player 1: " + str(score1)
        text = font.render(scoreprint, 1, WHITE)
        textpos = (0, 0)
        screen.blit(text, textpos)

        scoreprint = "Player 2: " + str(score2)
        text = font.render(scoreprint, 1, WHITE)
        textpos = (300, 0)
        screen.blit(text, textpos)

        # Draw Everything
        movingsprites.draw(screen)

        # Update the screen
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()


def create_player(game_type):
    if game_type == AUTOMATIC_PLAYER:
        return AutomaticPlayer(25, WHITE)
    return HumanPlayer(25, WHITE)


if __name__ == '__main__':
    pong()
