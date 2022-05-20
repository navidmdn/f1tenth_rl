import gym
import numpy as np
from scipy.spatial.distance import cdist
import pathlib
import os


class F110RaceEnv(gym.Env):
    def __init__(self, env_config,
                 test_map_name=None,
                 scan_range=10.0,
                 max_v=15.0,
                 n_cps=110,
                 cp_reward=0.1,
                 min_cp_dist=3.0,
                 maps_path=os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'maps'),
                 train_map_name='Oschersleben',
                 padding_punish=-0.05,
                 safe_padding=0.4,
                 laps=1.0,
                 min_deg=110,
                 max_deg=250,
                 begin_pos=[0.0702245, 0.3002981, 2.79787]
                 ):
        """
        This is a wrapper environment for the f1tenth gym environment
        params:
            env_config: configs of the environment required by rllib
            test_map_name: in test mode, use this map for evaluating agent
            scan_range: bounding laser scan in meters
            max_v: maximum velocity of the agent
            n_cps: number of checkpoints to extract from the waypoints file in equal distances
            cp_reward: the minimum reward agent receives for reaching checkpoints
            min_cp_dist: radius of each checkpoint
            maps_path: path to maps file
            train_map_name: base name of the training maps
            padding_punish: punishment that agent receives when it gets close to obstacles
            safe_padding: the padding distance from obstacles. if agent gets closer it gets punishment
            laps: number of laps to take before terminating
            min_deg: minimum degree of the lidar scan to use
            max_deg: maximum degree of the lidar scan to use
            begin_pos: position of the car in test mode
        """

        self.laps = laps
        self.maps_path = maps_path
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(142,), dtype=np.float32)
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
        self.padding_punish = padding_punish

        self.total_padding_punish = 0
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.next_cp_idx = 0
        self.prev_cp_t = 0
        self.begin_pos = begin_pos

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
        with open(f'{self.maps_path}/{self.train_map_name}{no}-wp.np', 'rb') as f:
            wps = np.load(f)

        return map_path, wps

    def load_test_setup(self):
        map_path = f"{self.maps_path}/{self.test_map_name}"
        with open(f'{self.maps_path}/{self.train_map_name}1-wp.np', 'rb') as f:
            wps = np.load(f)

        return map_path, wps

    def build_checkpoints(self, wps, cp_dist):
        """
        based on collected waypoints and the required distance between checkpoints, build checkpoints
        """
        start = np.random.randint(0, len(wps) - 1)
        checkpoints = []

        i = 0
        while i < len(wps):
            if i % cp_dist == 0:
                checkpoints.append(wps[(start + i) % len(wps)])
            i += 1
        return checkpoints

    def reset(self):
        self.prev_vel = [3.0] * 100
        self.total_padding_punish = 0
        self.cum_r = 0
        self.prev_cp_t =  0

        if self.test_map_name is not None:
            map_path, wps = self.load_test_setup()
        else:
            map_path, wps = self.load_train_setup()

        self.map_path = map_path
        self.env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1)
        self.env.add_render_callback(self._render_callback)

        cp_dist = len(wps) // self.n_cps

        self.checkpoints = self.build_checkpoints(wps, cp_dist)

        if self.test_map_name is None:
            # rotates the agent randomly to experience both directions
            rotate = np.random.choice([0, np.pi])
            if rotate != 0:
                self.checkpoints = self.checkpoints[::-1]

            start_idx = np.random.randint(0, len(self.checkpoints))
            start_point = self.checkpoints[start_idx]
            start_point[-1] += rotate

        else:
            dists = cdist([self.begin_pos[:2]], np.array(self.checkpoints)[:, :2])
            start_idx = np.argmin(dists)
            start_point = self.begin_pos

        # choose next checkpoint a few checkpoints forward to avoid initial unexpected reward
        self.next_cp_idx = (start_idx + 2) % len(self.checkpoints)

        obs, step_reward, done, info = self.env.reset(
            np.array([
                start_point
            ])
        )

        self.t = 0
        return self.to_vector_state(obs)

    def to_vector_state(self, obs, buck=3):
        ranges = obs['scans'][0]
        ranges = ranges[int((self.min_deg / 360) * len(ranges)):int((self.max_deg / 360) * len(ranges))]
        scanner = np.clip(ranges, 0, self.scan_range)

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

        # apply noise on observation for a more robust training
        if self.test_map_name is None:
            noise = np.random.normal(loc=0.0, scale=0.02, size=state.shape)
            state = state + noise
        return state

    def checkpoint(self, position, t):
        """
        rewards the agent for reaching each checkpoint
        current method rewards based on the time it takes to travel between each two checkpoints
        """

        # keep track of potential returnings
        return_cp_idx = (self.next_cp_idx - 5) % len(self.checkpoints)
        return_dist = np.linalg.norm(position - self.checkpoints[return_cp_idx][:2])
        dist = np.linalg.norm(position - self.checkpoints[self.next_cp_idx][:2])
        reward = 0
        if dist < self.min_cp_dist:
            time_diff = t - self.prev_cp_t

            # tune the reward according to the map
            reward = max(self.cp_reward + (0.4 - time_diff) * 0.5, self.cp_reward)
            self.prev_cp_t = t
            self.next_cp_idx = (self.next_cp_idx + 1) % len(self.checkpoints)

        elif return_dist < 3.0:
            reward = -1
        return reward

    def step(self, action: np.ndarray):
        reward = 0

        # we assume that the velocity given by the algorithm is in range (-1, 1)
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

        finish_time = 100
        if obs['collisions'][0] == 1.0:
            reward = -1
            done = True
            return next_state, reward, done, {'finish_time': finish_time}

        cp_reward = self.checkpoint(position, obs['lap_times'][0])
        reward += cp_reward
        self.t += 1

        if obs['lap_counts'][0] == self.laps:
            finish_time = obs['lap_times'][0]
            print(
                f"agent on {self.map_path} got reward {self.cum_r} in {finish_time}. Finish reward: {reward}. Padding punish: {self.total_padding_punish}")
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
            reward += self.padding_punish
            self.total_padding_punish += self.padding_punish

        self.cum_r += reward
        return next_state, reward, done, {'pos': position, 'finish_time': finish_time,
                                          'padding': self.total_padding_punish}

    def render(self):
        self.env.render()
