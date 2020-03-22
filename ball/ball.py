import math
import random

import pygame


class Ball(pygame.sprite.Sprite):
    def __init__(self, color, screen_width):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Create the image of the ball
        self.screen_width = screen_width
        self.image = pygame.Surface([10, 10])

        # Color the ball
        self.image.fill(color)

        # Get a rectangle object that shows where our image is
        self.rect = self.image.get_rect()

        # Get attributes for the height/width of the screen
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()

        # Speed in pixels per cycle
        self.speed = 0

        # Floating point representation of where the ball is
        self.x = 0
        self.y = 0

        # Direction of ball in degrees
        self.direction = 0

        # Height and width of the ball
        self.width = 10
        self.height = 10

        # Set the initial ball speed and position
        self.reset()

    def reset(self):
        self.x = random.randrange(50, self.screen_width - 50)
        self.y = 350.0
        self.speed = 8.0

        # Direction of ball (in degrees)
        self.direction = random.randrange(-45, 45)

        # Flip a 'coin'
        if random.randrange(2) == 0:
            # Reverse ball direction, let the other guy get it first
            self.direction += 180
            self.y = 50

    # This function will bounce the ball off a horizontal surface (not a vertical one)
    def bounce(self, diff):
        self.direction = (180 - self.direction) % 360
        self.direction -= diff

        if 75 < self.direction < 105:
            if self.direction > 90:
                self.direction = 105
            else:
                self.direction = 75

        if 255 < self.direction < 285:
            if self.direction > 270:
                self.direction = 285
            else:
                self.direction = 255

        # Speed the ball up
        self.speed *= 1.1

    # Update the position of the ball
    def update(self):
        # Sine and Cosine work in degrees, so we have to convert them
        direction_radians = math.radians(self.direction)

        # Change the position (x and y) according to the speed and direction
        self.x += self.speed * math.sin(direction_radians)
        self.y -= self.speed * math.cos(direction_radians)

        # Move the image to where our x and y are
        self.rect.x = self.x
        self.rect.y = self.y

        # Do we bounce off the left of the screen?
        if self.x <= 0:
            self.direction = (360 - self.direction) % 360
            # self.x=1

        # Do we bounce of the right side of the screen?
        if self.x > self.screenwidth - self.width:
            self.direction = (360 - self.direction) % 360
