import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        # x: [B, D]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(self.eps))
        self.count.copy_(tot_count)

    def forward(self, x, update=True):
        if self.training and update:
            self.update(x)
        std = torch.sqrt(self.var + self.eps)
        return (x - self.mean) / std, std

    def denorm(self, x_norm):
        std = torch.sqrt(self.var + self.eps)
        return x_norm * std + self.mean


class SkillDynamicsMoG(nn.Module):
    """
    q_phi(Δs | s, z) as a Mixture of Gaussians with K experts.
    - Predicts Δs
    - Input normalization for [s, z]
    - Target normalization for Δs
    - Optionally fixes covariance per component or learns diag std
    """
    def __init__(self, state_dim, skill_dim, hidden_dim, num_experts=4, learn_std=False):
        super().__init__()
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        self.hidden_dim = hidden_dim
        self.K = num_experts
        self.learn_std = learn_std

        # Normalizers
        self.in_norm = RunningNorm(state_dim + skill_dim)
        self.out_norm = RunningNorm(state_dim)

        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Heads
        self.logits_head = nn.Linear(hidden_dim, self.K)                 # mixture weights
        self.means_head = nn.Linear(hidden_dim, self.K * state_dim)      # K means
        if self.learn_std:
            self.logstd_head = nn.Linear(hidden_dim, self.K * state_dim) # K stds
        else:
            # Fixed std parameter per dimension per component
            self.logstd_param = nn.Parameter(torch.full((self.K, state_dim), -1.0))

    def _params(self, s, z, update_norm=True):
        x = torch.cat([s, z], dim=-1)
        x_norm, _ = self.in_norm(x, update=update_norm)
        h = self.trunk(x_norm)
        logits = self.logits_head(h)                       # [B, K]
        means = self.means_head(h).view(-1, self.K, self.state_dim)  # [B, K, D]

        if self.learn_std:
            logstd = self.logstd_head(h).view(-1, self.K, self.state_dim)
            logstd = torch.clamp(logstd, -5.0, 2.0)
        else:
            # Broadcast fixed logstd
            B = s.shape[0]
            logstd = self.logstd_param.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        return logits, means, logstd

    def get_log_prob(self, state, skill, next_state):
        # Δs (target) and normalization
        delta = next_state - state
        delta_norm, _ = self.out_norm(delta, update=self.training)

        logits, means_norm, logstd = self._params(state, skill, update_norm=self.training)
        std = logstd.exp()

        # Gaussian log-likelihood per component
        dist = Normal(means_norm, std)  # over normalized Δs
        # sum over state dims
        log_prob_comp = dist.log_prob(delta_norm.unsqueeze(1)).sum(dim=-1)  # [B, K]
        # mixture logsumexp
        log_mix = torch.logsumexp(logits + log_prob_comp, dim=-1)           # [B]
        # subtract logsumexp over logits for proper mixture prob normalization
        log_Z = torch.logsumexp(logits, dim=-1)
        return (log_mix - log_Z)  # [B]

    @torch.no_grad()
    def predict(self, state, skill):
        # Use mixture mean of normalized Δs, then de-normalize
        logits, means_norm, _ = self._params(state, skill, update_norm=False)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # [B, K, 1]
        mix_mean_norm = (weights * means_norm).sum(dim=1)  # [B, D]
        delta = self.out_norm.denorm(mix_mean_norm)
        return state + delta
    
class Actor(nn.Module):
    """Skill-conditioned policy π(a | s, z)"""
    def __init__(self, state_dim, skill_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

    def sample(self, state, skill):
        mean, log_std = self.forward(state, skill)
        dist = Normal(mean, log_std.exp())
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    """Twin Q-networks Q(s, z, a)"""
    def __init__(self, state_dim, skill_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + skill_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + skill_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, skill, action):
        x = torch.cat([state, skill, action], dim=-1)
        return self.q1(x), self.q2(x)
