import numpy as np
import gym
import time
import yaml
import gym
from argparse import Namespace
import os
import pathlib


class FGAgent:
    """
    a simple follow the gap agent
    """

    def __init__(self, bubble_r=0.5, lookahead_degrees=[90., 270.], speed=2.0):
        self.bubble_r = bubble_r
        self.lookahead_degrees = lookahead_degrees
        self.speed = speed

    def get_action(self, state):
        ranges = state[:360]
        # print(ranges[10], ranges[90], ranges[180], ranges[270], ranges[350])
        beg = int(self.lookahead_degrees[0])
        end = int(self.lookahead_degrees[1])

        ranges = ranges[beg:end]

        ranges = np.clip(ranges, 0, 10.0)

        nearest_point = np.min(ranges)
        min_dist = nearest_point + self.bubble_r

        ranges = np.where(ranges <= min_dist, 0.0, ranges)

        gaps = []
        if ranges[0] != 0.0:
            gaps.append(0)

        i = 0
        while i < len(ranges):
            if ranges[i] == 0.0:
                if i > 0:
                    gaps.append(i - 1)
                while i < len(ranges) and ranges[i] == 0.0:
                    i += 1
                if i < len(ranges):
                    gaps.append(i)
                continue

            i += 1

        if ranges[-1] != 0.0:
            gaps.append(len(ranges) - 1)

        assert len(gaps) % 2 == 0

        max_gap = -1
        gap_beg = 0
        gap_end = 360

        # find max gap
        i = 0
        while i < len(gaps):
            if gaps[i + 1] - gaps[i] > max_gap:
                max_gap = gaps[i + 1] - gaps[i]
                gap_beg = gaps[i] + beg
                gap_end = gaps[i + 1] + beg
            i += 2

        mid_point = float(gap_end + gap_beg) / 2.
        angle_deg = mid_point - 180
        angle_rad = angle_deg * (np.pi / 180.)
        #         print(f"midpoint:{mid_point}")
        angle = angle_rad * 1.0
        return [angle, self.speed]


class WPEnv(gym.Env):

    def __init__(self, map_name='', begin_pos=None):
        self.map_name = map_name
        self.begin_pos = begin_pos
        self.reset()

    def reset(self):
        def render_callback(env_renderer):
            # custom extra drawing function

            e = env_renderer

            # update camera to follow car
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left = left - 800
            e.right = right + 800
            e.top = top + 800
            e.bottom = bottom - 800

        map_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'maps', self.map_name)

        self.env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
        self.env.add_render_callback(render_callback)

        begin_pos = [0, 0, 0] if self.begin_pos is None else self.begin_pos
        obs, step_reward, done, info = self.env.reset(np.array([begin_pos]))

        return self.to_vector_state(obs)

    def to_vector_state(self, obs):

        scanner = np.clip(obs['scans'][0], 0, 10)
        buck = 3
        size = 1080 // buck
        agg_scanner = np.zeros(size, )
        for i in range(size):
            agg_scanner[i] = np.mean(scanner[i * buck: i * buck + buck])

        state = [
            agg_scanner,
            np.array(obs['linear_vels_x'][0]),
            np.array(obs['ang_vels_z'][0]),
            (obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])
        ]

        return state

    def step(self, action):
        act = np.array([action])
        reward = 0
        obs, step_reward, done, info = self.env.step(act)

        if obs['collisions'][0] == 1.0:
            print("CRASHED")
            done = True
            reward = -1

        if obs['lap_counts'][0] == 1.0:
            reward = 1.0
            print("DONE")
            done = True

        next_state = self.to_vector_state(obs)
        return next_state, reward, done, info

    def render(self):
        self.env.render()


if __name__ == '__main__':
    """
    You can run this script to build waypoints for a specific map using a simple "follow the gap" agent
    """

    done = False
    map_name = 'race2'
    env = WPEnv(map_name=map_name, begin_pos=[0.0702245, 0.3002981, 2.79787])
    obs = env.reset()
    agent = FGAgent(speed=3.0, bubble_r=0.5)

    positions = []

    while not done:
        positions.append(obs[3])
        th, v = agent.get_action(obs[0])
        obs, r, done, _ = env.step([th, v])
        env.render()

    path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'maps', f"{map_name}-wp.np'")
    with open(path, 'wb') as f:
        np.save(f, positions)
