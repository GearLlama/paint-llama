#!/usr/bin/env python3
import random
import os
import time
from typing import Optional
import numpy as np
import torch
from rl_painter.evaluator import Evaluator
from rl_painter.tensorboard import Writer
from rl_painter.environment import FastPaintEnvironment
from rl_painter.agent import AgentDDPG

from rl_painter.utils import prRed, prBlack

BASE_DIR = "/mnt/f/paint_llama/rl_painter"
LOGS_DIR = f"{BASE_DIR}/logs/train/{str(int(time.time()))}"
PKLS_DIR = f"{BASE_DIR}/pkls/{str(int(time.time()))}"

WARMUP = 400
DISCOUNT = 0.95**5
BATCH_SIZE = 96
RMSIZE = 800
ENV_BATCH = 96
TAU = 0.001
MAX_STEP = 40
NOISE_FACTOR = 0
VALIDATE_INTERVAL = 50
VALIDATE_EPISODES = 5
TRAIN_TIMES = 2000000
EPISODE_TRAIN_TIMES = 10
RESUME: Optional[str] = None
DEBUG = True
SEED = 1234


def train(agent: AgentDDPG, env: FastPaintEnvironment, evaluator: Evaluator, writer: Writer) -> None:
    train_times = TRAIN_TIMES
    noise_factor = NOISE_FACTOR
    max_step = MAX_STEP
    warmup = WARMUP
    validate_interval = VALIDATE_INTERVAL
    debug = DEBUG
    episode_train_times = EPISODE_TRAIN_TIMES

    # Initialize
    time_stamp = time.time()
    step = episode = episode_steps = 0
    observation = None

    # Start training
    while step <= train_times:
        print("running step ", step)
        step += 1
        episode_steps += 1

        # Reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        # Select action and observe
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done)

        if episode_steps >= max_step and max_step:
            if step > warmup:
                # Evaluate and save model
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluator(env, agent.select_action, debug=debug)
                    if debug:
                        prRed(
                            f"Step_{(step - 1):07d}: mean_reward:{np.mean(reward):.3f} mean_dist:{np.mean(dist):.3f} var_dist:{np.var(dist):.3f}"
                        )
                    writer.add_scalar("validate/mean_reward", np.mean(reward), step)
                    writer.add_scalar("validate/mean_dist", np.mean(dist), step)
                    writer.add_scalar("validate/var_dist", np.var(dist), step)
                    agent.save_model(PKLS_DIR)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.0
            tot_value_loss = 0.0

            if step > warmup:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for _ in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar("train/critic_lr", lr[0], step)
                writer.add_scalar("train/actor_lr", lr[1], step)
                writer.add_scalar("train/Q", tot_Q / episode_train_times, step)
                writer.add_scalar("train/critic_loss", tot_value_loss / episode_train_times, step)

            if debug:
                prBlack(
                    f"#{episode}: steps:{step} interval_time:{train_time_interval:.2f} train_time:{(time.time() - time_stamp):.2f}"
                )
            time_stamp = time.time()

            # reset
            observation = None
            episode_steps = 0
            episode += 1


if __name__ == "__main__":
    # Make output and pkl dirs
    os.makedirs(PKLS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create writer
    writer = Writer(LOGS_DIR)

    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Create environment, agent and evaulator
    fenv = FastPaintEnvironment(MAX_STEP, ENV_BATCH, writer)
    agent = AgentDDPG(
        BATCH_SIZE,
        ENV_BATCH,
        MAX_STEP,
        TAU,
        DISCOUNT,
        RMSIZE,
        writer,
        RESUME,
    )
    evaluator = Evaluator(VALIDATE_EPISODES, MAX_STEP, ENV_BATCH, writer)

    # Train
    print("observation_space", fenv.observation_space, "action_space", fenv.action_space)
    train(agent, fenv, evaluator, writer)
