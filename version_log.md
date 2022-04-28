## v1.0

### environment configs

scan range: 10m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.1
max_v: 12
noise on obs: N(0, 0.03)
obs dim: 110


### trainig config

alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

slow:
- solves train1: 126.95s
- solves train2: 138.2s
fails on obs


## v1.1

### changelog

increased cp reward

### environment configs

scan range: 10m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.5
max_v: 12
noise on obs: N(0, 0.03)
obs dim: 110


### trainig config

alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

- solves train1: 82s
- fails train2
- fails on obs


## v1.2

### changelog

only trained on train2

### environment configs

scan range: 10m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.5
max_v: 12
noise on obs: N(0, 0.03)
obs dim: 110


### trainig config

alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

- solves train2: 96s
- solves obs: 93s
- fails on train1


## v1.3

### changelog

scan range to 15

### environment configs

scan range: 15m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.5
max_v: 12
noise on obs: N(0, 0.03)
obs dim: 110


### trainig config

alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

awful!


## v1.4

### changelog

reduced speed to 10
no noise
gamma increased

### environment configs

scan range: 10m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.5
max_v: 10
obs dim: 110


### trainig config

gamma: 0.995
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0001

### result analysis

worked on train1,2
fails on obs


## v1.5

### changelog

reduced speed to 10
add noise to obs
gamma increased
padding 15cm

### environment configs

scan range: 10m
maps: ss train 1,2 
wps: 101 points ~ 3m apart
cp radius: 3m
cp reward: 0.1
max_v: 10
obs dim: 110


### trainig config

gamma: 0.995
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

Doesn't solve any!


## v1.6

### changelog

padding 5cm
added third training track
added a reward for fast completion (exp(4-t/20)) -> 80s gets a reward of e 100s gets a reward of 1/e

### environment configs

scan range: 10m
maps: ss train 1,2,3
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.1
max_v: 10
obs dim: 110


### trainig config

gamma: 0.99
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003
padding: 5cm

### result analysis

gets stuck in train3
solves all others

## v1.7

### changelog

stopping and returning punishments

### environment configs

scan range: 10m
maps: ss train 1,2,3
wps: 101 points ~ 3m apart
cp radius: 2m
cp reward: 0.1
max_v: 10
obs dim: 110


### trainig config

gamma: 0.99
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003
padding: 5cm

### result analysis

works on 1, 2
fails on 3
solves test sometimes!


## v1.8

### changelog

less workers
only train on 1,2
trimmed obs to only look forward 70-290 deg

### environment configs

scan range: 10m
maps: ss train 1,2
wps: 200 points 
cp radius: 3m
cp reward: 0.1
max_v: 10
obs dim: 220


### trainig config

gamma: 0.99
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003
padding: 5cm

### result analysis

solves all begining from start point


## v1.9

### changelog

increased speed to 12

### environment configs

scan range: 10m
maps: ss train 1,2
wps: 200 points 
cp radius: 2m
cp reward: 0.1
max_v: 12
obs dim: 220


### trainig config

gamma: 0.99
alg: ppo
num_workers: 16
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003
padding: 5cm

### result analysis

solves train1,2,3
fails at obs


## v2.0*

### changelog

padding of 30 cm

### environment configs

scan range: 10m
maps: ss train 1,2
wps: 200 points 
cp radius: 2m
cp reward: 0.1
max_v: 12
obs dim: 220
padding: 30cm neg reward
finish time reward: max(0.2*(50 - t), 0.1)


### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003


### result analysis

checkpoint 36
solves train1,2,3
solves obs

## v2.1

### changelog

increased speed to 15

### environment configs

scan range: 10m
maps: ss train 1,2
wps: 200 points 
cp radius: 2m
cp reward: 0.1
max_v: 15
obs dim: 220
padding: 30cm neg reward
finish time reward: max(0.2*(50 - t), 0.1)


### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003


### result analysis

can't solve all


## v2.2

### changelog

more complex network

### environment configs

scan range: 10m
maps: ss train 1,2,3
wps: 200 points 
cp radius: 3m
cp reward: 0.1
max_v: 12
obs dim: 220
padding: 30cm neg reward
finish time reward: max(0.2*(50 - t), 0.1)


### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
network: (300, 300, 300)
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 16
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003


### result analysis

solves all. a bit slower

## v2.3*

### changelog

padding of 30 cm

### environment configs

scan range: 10m
maps: ss train 1,2,3
wps: 200 points 
cp radius: 2m
cp reward: 0.1
max_v: 12
obs dim: 110
padding: 30cm neg reward
finish time reward: 0.1


### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 1
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003


### result analysis



## sim-v1.0


### environment configs

scan range: 10m
maps: train 1
wps: 100 points 
cp radius: 3m
cp reward: 0.1
max_v: 12
obs dim: 222
obs range: 70 to 290
padding: 30cm neg reward 0.05
finish time reward:  
max(0.2*(50 - t), self.cp_reward)

### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 1
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

solved both train and test in ~46s
had room for improvement (only 50 iters)


## sim-v1.0.1

### changelog
fine tuning v1.0
changed final reward a bit

### environment configs

scan range: 10m
maps: train 1
wps: 100 points 
cp radius: 3m
cp reward: 0.1
max_v: 12
obs dim: 222
obs range: 70 to 290
padding: 30cm neg reward 0.05
finish time reward:  
max(0.2*(45 - t), self.cp_reward)

### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 1
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis
last checkpoint super fast but fails at test
cp -> need to have an evaluation env and another test env


## phy-v1.0.5


### environment configs

scan range: 10m
maps: train 1
wps: 100 points 
cp radius: 3m
cp reward: 0.1
max_v: 10
obs dim: 182
obs range: 90 to 180
padding: 30cm neg reward 0.05
finish time reward:  
max(0.2*(45 - t), self.cp_reward)

### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 1
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis

successful first try
some points crash to obstacles
in turns so close to wall
low steering angle on car


## phy-v1.0.6

### changelog

angle reduced to 0.34
noise var to 0.1


### environment configs

scan range: 10m
maps: train 1
wps: 100 points 
cp radius: 3m
cp reward: 0.1
max_v: 10
obs dim: 182
obs range: 90 to 180
padding: 30cm neg reward 0.05
finish time reward:  
max(0.2*(45 - t), self.cp_reward)

### trainig config

gamma: 0.99
alg: ppo
num_workers: 15
num_gpus: 1.0
kl_coeff: 1.0
clip_param: 0.2
num_envs_per_worker: 1
train_batch_size: 100000
sgd_minibatch_siz': 4096
batch_mode: 'truncate_episodes'
lr: .0003

### result analysis
