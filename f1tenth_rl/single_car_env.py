import gym
import numpy as np
import time
import yaml
from scipy.spatial.distance import cdist


class F110RaceEnv(gym.Env):
    def __init__(self, env_config,
                 test_map_name=None,
                 scan_range=10.0,
                 max_v=12.0,
                 n_cps=200,
                 cp_reward=0.1,
                 min_cp_dist=3.0,
                 maps_path='./maps',
                 train_map_name='SILVERSTONE_TRAIN',
                 safe_padding=0.3,
                 laps=1.0,
                 return_cp_dist=3.0,
                 num_cp_to_consider_return=5,
                 in_padding_punish=0.05
                 ):

        self.laps = laps
        self.maps_path = maps_path
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(110,), dtype=np.float32)
        self.max_v = max_v
        self.test_map_name = test_map_name
        self.min_cp_dist = min_cp_dist
        self.n_cps = n_cps
        self.cp_reward = cp_reward
        self.scan_range = scan_range
        self.safe_padding = safe_padding
        self.map_path = None
        self.train_map_name = train_map_name
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.cum_r = 0
        self.prev_vel = [4.0] * 100
        self.total_padding_punish = 0
        self.return_cp_dist = return_cp_dist
        self.num_cp_to_consider_return = num_cp_to_consider_return
        self.in_padding_punish = in_padding_punish

    def _render_callback(self, env_renderer):
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

    def load_train_setup(self):
        no = np.random.choice([1, 2, 3])

        map_path = f"{self.maps_path}/{self.train_map_name}{no}"
        with open(f'{self.maps_path}/ss{no}-wp.np', 'rb') as f:
            wps = np.load(f)

        return map_path, wps

    def load_test_setup(self):
        map_path = f"{self.maps_path}/{self.test_map_name}"
        with open(f'{self.maps_path}/ss-wp.np', 'rb') as f:
            wps = np.load(f)

        return map_path, wps

    def build_checkpoints(self, wps, cp_dist):

        start = np.random.randint(0, len(wps) - 1)
        checkpoints = []

        i = 0
        while i < len(wps):
            if i % cp_dist == 0:
                checkpoints.append(wps[(start + i) % len(wps)])
            i += 1
        return checkpoints

    def reset(self):
        self.prev_vel = [4.0] * 100
        self.total_padding_punish = 0
        self.cum_r = 0

        if self.test_map_name is not None:
            map_path, wps = self.load_test_setup()
        else:
            map_path, wps = self.load_train_setup()

        #         print(map_path, len(wps))
        self.map_path = map_path
        self.env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
        self.env.add_render_callback(self._render_callback)

        cp_dist = len(wps) // self.n_cps

        self.checkpoints = self.build_checkpoints(wps, cp_dist)
        #         print(f"number of checkpoints: {len(self.checkpoints)}")

        if self.test_map_name is None:
            start_idx = np.random.randint(0, len(self.checkpoints))
            start_point = self.checkpoints[start_idx]

        else:
            dists = cdist([[0, 0]], np.array(self.checkpoints)[:, :2])
            start_idx = np.argmin(dists)
            start_point = [0., 0., np.pi + 1]

        self.next_cp_idx = (start_idx + 2) % len(self.checkpoints)

        obs, step_reward, done, info = self.env.reset(
            np.array([
                start_point
            ])
        )

        self.t = 0
        return self.to_vector_state(obs)

    def to_vector_state(self, obs):

        scanner = np.clip(obs['scans'][0], 0, self.scan_range)

        buck = 10
        size = len(scanner) // buck
        agg_scanner = np.zeros(size, )
        for i in range(size):
            agg_scanner[i] = np.min(scanner[i * buck: i * buck + buck])

        agg_scanner /= self.scan_range
        state = np.concatenate([
            agg_scanner,
            np.array(obs['linear_vels_x'][:1]) / self.max_v,
            np.array(obs['ang_vels_z'][:1]) / self.max_v,
        ])

        #         if self.test_map_name is None:
        #             noise = np.random.normal(loc=0.0, scale=0.03, size=state.shape)
        #             state = state + noise

        return state

    def checkpoint(self, position):
        return_cp_idx = (self.next_cp_idx - 1 - self.num_cp_to_consider_return) % len(self.checkpoints)
        return_dist = np.linalg.norm(position - self.checkpoints[return_cp_idx][:2])
        dist = np.linalg.norm(position - self.checkpoints[self.next_cp_idx][:2])
        reward = 0
        if dist < self.min_cp_dist:
            reward = self.cp_reward
            self.next_cp_idx = (self.next_cp_idx + 1) % len(self.checkpoints)
        elif return_dist < self.return_cp_dist:
            reward = -1
        return reward

    def step(self, action):
        reward = 0

        act_v = action[0] * (self.max_v / 2) + (self.max_v / 2)
        self.prev_vel.insert(0, act_v)
        self.prev_vel.pop()

        th = action[1] * np.pi / 6
        act = np.array([[th, act_v]])

        obs, step_reward, done, info = self.env.step(act)
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]

        next_state = self.to_vector_state(obs)
        position = np.array([pose_x, pose_y])

        if obs['collisions'][0] == 1.0:
            reward = -1
            done = True
            if self.test_map_name is not None:
                print('CRASHED')
            return next_state, reward, done, {}

        cp_reward = self.checkpoint(position)
        reward += cp_reward
        self.t += 1

        if obs['lap_counts'][0] == self.laps:
            t = obs['lap_times'][0]
            #             reward = np.exp(2.5 - 0.05*t)
            #             fairlap = 60
            #             reward = max(0.2*(fairlap - t), self.cp_reward)
            reward = self.cp_reward
            print(
                f"agent on {self.map_path} got reward {self.cum_r} in {t}. Finish reward: {reward}. Padding punish: {self.total_padding_punish}")
            done = True
        elif cp_reward == -1:
            t = obs['lap_times'][0]
            print(f"agent on {self.map_path} got reward {self.cum_r} in {t}. RETURNED")
            done = True
        elif np.max(self.prev_vel) < 3.0:
            t = obs['lap_times'][0]
            print(f"agent on {self.map_path} got reward {self.cum_r} in {t}. STOPPED")
            reward = -1
            done = True

        if done and self.test_map_name is not None:
            print('lap:', obs['lap_times'])

        if np.min(obs['scans'][0]) < self.safe_padding:
            reward += self.in_padding_punish
            self.total_padding_punish += self.in_padding_punish

        self.cum_r += reward
        return next_state, reward, done, {'pos': position}

    def render(self):
        self.env.render()
