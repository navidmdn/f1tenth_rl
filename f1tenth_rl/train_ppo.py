import ray
from ray.rllib.agents import ppo
from f1tenth_rl.single_car_env import F110RaceEnv

import numpy as np
import yaml
import os
import argparse

from torch.utils.tensorboard import SummaryWriter


def evaluate(model, envs):
    rewards = []

    for env in envs:
        done = False
        obs = env.reset()

        r = 0
        while not done:
            action = model.compute_single_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            r += reward

        rewards.append(r)

    return np.mean(rewards)


if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="single_car", help='path to yaml config file')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--verbose', default=5, help='number of epochs to train')
    parser.add_argument('--logname', default="ppo_single_car", help='number of epochs to train')

    args = parser.parse_args()
    writer = SummaryWriter(os.path.join("runs", args.logname))

    with open(os.path.join('f1tenth_rl', 'configs', f"{args.config}.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    trainer = ppo.PPOTrainer(env=F110RaceEnv, config=config['trainer'])

    # print("FINE TUNING")
    # trainer.restore('../checkpoints/v2.2/checkpoint_000031/checkpoint-31')

    print(trainer.config)
    rewards = []
    best_reward = -100
    models_path = './checkpoints'

    eval_envs = [
        F110RaceEnv({}, test_map_name='race_test1', laps=1),
        F110RaceEnv({}, test_map_name='race4', laps=1),
        F110RaceEnv({}, test_map_name='race5', laps=1),
        F110RaceEnv({}, test_map_name='race6', laps=1),
    ]

    sim_name = args.config
    eval_rewards = []

    for i in range(args.epochs):
        result = trainer.train()
        episode_r = result['episode_reward_mean']
        print(f"episode: {i} reward:{episode_r}")
        writer.add_scalar("reward/train", episode_r, i)

        if i % args.verbose == 0:
            eval_reward = evaluate(trainer, envs=eval_envs)
            writer.add_scalar("reward/eval", eval_reward, i)

            if eval_reward > best_reward:
                best_reward = eval_reward
                cp = trainer.save(f"{models_path}/{sim_name}")
                print("checkpoint saved at", cp)

    writer.flush()
    writer.close()
