import pygame
import time

import cv2
import numpy as np

from snake import snake
from brain import brain


def init_game(w, h, w_input, h_input, load = True):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("snake ai")

    screen = pygame.display.set_mode((w, h))
    game = snake(w, h)
    game_brain = brain(w_input, h_input)
    if load:
        game_brain.load()

    return screen, game, game_brain


def update_window(screen, game):
    screen.fill((0, 0, 0))
    game.draw(screen)

    pygame.display.update()


def preprocess(state, w, h, top_padding, w_input, h_input):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)  # greyscale
    state = state[top_padding:h, 0:w]  # crop
    state = cv2.resize(state, (w_input, h_input))  # resize

    show(state)

    # expand dims for tensorflow model
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=4)
    return state


def show(state):
    cv2.imshow('image', state)


def main():
    w = 400
    h = int(w * 1.1)
    w_input = h_input = 84
    screen, game, game_brain = init_game(w, h, w_input, h_input)

    timestep = 0
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
                    timestep = max(0, timestep - 0.1)
                if event.key == pygame.K_DOWN:
                    timestep = timestep + 0.1
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_s:
                    game_brain.save()
                if event.key == pygame.K_l:
                    game_brain.load()

        if paused:
            continue

        update_window(screen, game)

        # get state
        # TODO - 4 state history stack
        state_a, reward = game.get_state(screen)
        state_a = preprocess(state_a, w, h, h - w, w_input, h_input)
        # show(state_a) #debug

        # generate action
        action = game_brain.think(state_a)

        # perform action and get new state
        game.perform_action(action)
        update_window(screen, game)
        state_b, reward = game.get_state(screen)
        state_b = preprocess(state_b, w, h, h - w, w_input, h_input)
        ended = game.is_ended()

        # brain stuff
        game_brain.short_memory_training(state_a, action, reward, state_b, ended)
        game_brain.remember(state_a, action, reward, state_b, ended)

        if ended:
            print(f"Round {game.round} | Score: {game.score} | Replay batch: {game_brain.mini_batch_size} | rand_thresh: {game_brain.rand_thresh}")
            game.reset(5)
            game_brain.short_memory_training(state_a, action, reward, state_b, ended, True)
            game_brain.long_memory_training()
            game_brain.save()

        if timestep > 0:
            time.sleep(timestep)


if __name__ == "__main__":
    main()
