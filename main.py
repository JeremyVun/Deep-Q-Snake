from collections import deque
import time

from snake import create_game
from brain import create_brain
from util import *


def main():
    # config
    conf_g, conf_b = read_config()
    w = int(conf_g['window_size'])
    h = int(w * 1.1)
    downsample_size = int(conf_b['input_size']) # downsample dimensions
    n_frames = int(conf_b['frame_buffer']) # temporal degree
    frame_hist = deque([]) # temporal frame buffer
    game_y = h - w

    # game vars
    screen, game = create_game(w, h, conf_g)
    game_brain = create_brain(conf_b)

    timestep = int(conf_g['timestep'])
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

        # maintain frame state stack
        while len(frame_hist) < n_frames:
            state, reward = game.get_state(screen)
            state = preprocess(state, game_y, downsample_size)
            frame_hist = queue(state, frame_hist, n_frames)

        state_a = pack(frame_hist)
        show(frame_hist)

        action = game_brain.think(state_a)
        game.perform_action(action)

        #get action result
        update_window(screen, game)
        state, reward = game.get_state(screen)
        state = preprocess(state, game_y, downsample_size)

        frame_hist.popleft()
        frame_hist = queue(state, frame_hist, n_frames)

        state_b = pack(frame_hist)
        show(frame_hist)
        ended = game.is_ended()

        # brain stuff
        game_brain.remember(state_a, action, reward, state_b, ended)

        if ended:
            print(f"Round {game.get_round()} | Score: {game.get_score()} | Replay batch: {game_brain.get_batch_size()}[{game_brain.get_epochs()}] | rand_thresh: {game_brain.get_action_thresh()}(10)")
            game.reset()
            game_brain.short_memory_training(state_a, action, reward, state_b, ended)
            game_brain.long_memory_training()
            frame_hist = deque([])
        else:
            game_brain.short_memory_training(state_a, action, reward, state_b, ended)

        if timestep > 0:
            time.sleep(timestep)


if __name__ == "__main__":
    main()
