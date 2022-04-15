import ray
from ray.rllib.agents import ppo
from f1tenth_rl.single_car_env import F110RaceEnv
from ray.tune.registry import register_env
import yaml
import os
import argparse
import pickle

if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="sim_v2.0", help='path to yaml config file')
    parser.add_argument('--epochs', default=100, help='number of epochs to train')
    args = parser.parse_args()

    with open(os.path.join('f1tenth_rl', 'configs', f"{args.config}.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    print(config['env'], config['trainer'])
    register_env("singleCarEnv", lambda cfg: F110RaceEnv(cfg, **config['env']))
    trainer = ppo.PPOTrainer(env='singleCarEnv', config=config['trainer'])

    # print("FINE TUNING")
    # trainer.restore('../checkpoints/v2.2/checkpoint_000031/checkpoint-31')

    print(trainer.config)
    rewards = []
    best_reward = -100
    models_path = './checkpoints'

    for i in range(args.epochs):
        result = trainer.train()
        episode_r = result['episode_reward_mean']
        print(f"episode: {i} reward:{episode_r}")
        rewards.append(episode_r)
        with open(f'{models_path}/{args.config}_r', 'wb') as f:
            pickle.dump(rewards, f)

        if i % 5 == 0 and episode_r > best_reward:
            best_reward = episode_r
            cp = trainer.save(f"{models_path}/{args.config}")
            print("checkpoint saved at", cp)
