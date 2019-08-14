# 1. code snake game
# 2. expose action interface
# 3. state output
# 4. implement dqn agent
# - random action based on how many games its played. more played = less random. random or use dll model
# - do move, return new state
# - set reward
# - s, a, r, s'
# - bellman function to calc loss and fit model. q(s,a) = r + ymax(q(s',a')). 
# - Loss = reward0 + discount * prediction(next state) - prediction(current state)
# - approximate action value function using bellman func. simulation of actual play in emulator prunes a lot of states
# - state -> model -> q value per action

# cost function
# target - actual
# gradient vector * -learning rate = delta v

import pygame
import time
from secrets import randbelow
from snake import snake
from brain import brain

import cv2
import numpy as np
import matplotlib.pyplot as plt


def init_game(w, h):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("snake ai")

    screen = pygame.display.set_mode((w, h))
    game = snake(w, h)
    game_brain = brain()

    return screen, game, game_brain


def update_window(screen, game):
    screen.fill((0, 0, 0))
    game.draw(screen)

    pygame.display.update()


def preprocess(state, w, h, top_padding):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) #greyscale
    state = state[top_padding:h, 0:w] #crop
    return cv2.resize(state, (200, 200)) #resize


def show(state):
    cv2.imshow('image', state)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    w = 400
    h = int(w * 1.1)
    screen, game, game_brain = init_game(w, h)
    rand_thresh = 5

    timestep = 1.0 / 5.0
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_UP:
                    timestep = max(0, timestep / 1.5)
                if event.key == pygame.K_DOWN:
                    timestep = timestep * 1.5
                if event.key == pygame.K_p:
                    paused = not paused

        if paused:
            continue

        update_window(screen, game)

        # get state
        state_a, reward = game.get_state(screen)
        state_a = preprocess(state_a, w, h, h - w)
        #show(state_a)

        # generate action
        if randbelow(10) > rand_thresh:
            action = game.random_action()
        else:
            action = game.to_action(game_brain.think(state_a))

        # perform action and get new state
        print("action: ", action)
        game.perform_action(action)
        update_window(screen, game)
        state_b, reward = game.get_state(screen)
        state_b = preprocess(state_b, w, h, h-w)
        ended = game.is_ended()

        #brain stuff
        game_brain.short_memory_training(state_a, action, reward, state_b, ended)
        game_brain.remember(state_a, action, reward, state_b, ended)

        if ended:
            game.reset(5)
            game_brain.long_memory_training()

        time.sleep(timestep)


# - s, a, r, s'
# - bellman function to calc loss and fit model. q(s,a) = r + ymax(q(s',a')).

if __name__ == "__main__":
    main()
