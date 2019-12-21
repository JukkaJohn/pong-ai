import pygame


class HumanAgent:
    
    def get_direction(self, key_state, ball_x, ball_y, own_player_x):
        return key_state[pygame.K_x] - key_state[pygame.K_z]
