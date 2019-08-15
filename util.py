import cv2
import pygame
import numpy as np
import configparser


def update_window(screen, game):
    screen.fill((0, 0, 0))
    game.draw(screen)

    pygame.display.update()


def pack(frame_hist, j = 0, expand = True):
    # (1, frames, dim, dim, 1)
    if expand:
        result = np.expand_dims(frame_hist[0], axis=j)
    else:
        result = frame_hist[0]

    for i in range(1, len(frame_hist)):
        if expand:
            new_frame = np.expand_dims(frame_hist[i], axis=j)
        else:
            new_frame = frame_hist[i]

        result = np.concatenate((result, new_frame), axis=j)

    if expand:
        result = np.expand_dims(result, axis=0)

    return result


def preprocess(state, y_start, downsample_size):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)  # greyscale
    state = state[y_start:, 0:]  # crop
    state = cv2.resize(state, (downsample_size, downsample_size))  # resize

    state = np.expand_dims(state, axis=4)

    return state


def show(frame_hist):
    frame = pack(frame_hist, 1, False)
    cv2.imshow('image', frame)


def debug_show(state):
    cv2.imshow('image', state)
    cv2.waitKey(0)


def queue(state, state_q, n):
    if len(state_q) >= n:
        state_q.popleft()
    state_q.append(state)
    return state_q


def read_config():
    config = configparser.ConfigParser()
    return config['GAME'], config['BRAIN']