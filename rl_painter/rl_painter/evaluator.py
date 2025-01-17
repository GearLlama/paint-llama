import numpy as np


class Evaluator:
    def __init__(self, validate_episodes: int, max_step: int, env_batch: int, writer) -> None:
        self.validate_episodes = validate_episodes
        self.max_step = max_step
        self.env_batch = env_batch
        self.writer = writer
        self.log = 0

    def __call__(self, env, policy, debug=False):
        # TODO: Add type hints and deal with unbound variables
        observation = None
        for episode in range(self.validate_episodes):
            # reset at the start of episode
            observation = env.reset(test=True, episode=episode)
            episode_steps = 0
            episode_reward = 0.0
            assert observation is not None
            # start episode
            episode_reward = np.zeros(self.env_batch)
            while episode_steps < self.max_step or not self.max_step:
                action = policy(observation)
                observation, reward, _, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                env.save_image(self.log, episode_steps)
            dist = env.get_dist()
            self.log += 1
        return episode_reward, dist
