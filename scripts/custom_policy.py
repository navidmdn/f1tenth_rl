import torch
import os

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=110, out_features=256, bias=True),
            torch.nn.Tanh()
        )
        self.hidden2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=256, bias=True),
            torch.nn.Tanh()
        )
        self.logits = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=4, bias=True),
        )
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.logits(x)

class PolicyWrapper:
    def __init__(self, load_path='models/'):
        self.device = torch.device('cuda:0')
        self.policy = Policy()
        self.policy.to(self.device)
        self.policy.hidden1.load_state_dict(torch.load(os.path.join(load_path, 'h1_sd')))
        self.policy.hidden2.load_state_dict(torch.load(os.path.join(load_path, 'h2_sd')))
        self.policy.logits.load_state_dict(torch.load(os.path.join(load_path, 'logit_sd')))
        
    def get_action(self, observation):
        obs_t = torch.tensor(observation, dtype=torch.float32).to(self.device)
        acts = torch.clip(self.policy(obs_t), min=-1, max=1).detach().cpu().numpy()
        v, th = acts[0], acts[1]
        
        return v, th