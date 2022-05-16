import gym
import numpy as np
from ray.rllib.agents import ppo
from f1tenth_rl.single_car_env import F110RaceEnv
import pathlib
import os


if __name__ == '__main__':
    evaluation_config = {
        'framework': 'torch',
        'model': {
            'fcnet_hiddens': [300, 300]
        },
        'observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(142,), dtype=np.float32),
        'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    }

    cp_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'checkpoints',
                           "race_v2.0.4/checkpoint_000281/checkpoint-281")
    agent = ppo.PPOTrainer(config=evaluation_config)
    agent.restore(cp_path)

    done = False
    env = F110RaceEnv({}, test_map_name='Oschersleben2', laps=1)
    obs = env.reset()

    r = 0
    while not done:
        action = agent.compute_single_action(obs, explore=False)
        obs, reward, done, info = env.step(action)
        env.render()
        r += reward

    print(f'reward:{r}')
    env.close()
