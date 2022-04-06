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
