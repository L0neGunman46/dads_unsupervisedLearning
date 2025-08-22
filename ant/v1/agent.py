import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Actor, Critic, SkillDynamicsMoG


class DADSAgent:
    def __init__(self, filtered_state_dim, full_state_dim, action_dim, config):
        self.config = config
        self.device = config.DEVICE
        self.skill_dim = config.NUM_SKILLS

        # Actor/Critic use FILTERED states (exclude x or x,y)
        self.actor = Actor(filtered_state_dim, self.skill_dim, action_dim, config.HIDDEN_DIM).to(
            self.device
        )
        self.critic = Critic(filtered_state_dim, self.skill_dim, action_dim, config.HIDDEN_DIM).to(
            self.device
        )
        self.critic_target = Critic(
            filtered_state_dim, self.skill_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)

        # Dynamics uses FULL states (to predict Δx, Δy, etc.)
        self.dynamics = SkillDynamicsMoG(
            full_state_dim,
            self.skill_dim,
            config.HIDDEN_DIM,
            num_experts=getattr(config, "DYNAMICS_NUM_EXPERTS", 4),
            learn_std=getattr(config, "DYNAMICS_LEARN_STD", False),
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.AGENT_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.AGENT_LR)
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=config.DYNAMICS_LR)

        # Fixed SAC alpha (optional: add auto-tuning later)
        self.alpha = torch.tensor(self.config.AGENT_ALPHA, device=self.device)

    def select_action(self, state, skill):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        skill = torch.as_tensor(skill, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(state, skill)
        return action.detach().cpu().numpy()[0]

    def _calculate_intrinsic_reward(
        self, state_full, skill, next_state_full, num_prior_samples=500
    ):
        """
        r = log q_phi(s'|s,z) - log E_{z'~p(z)}[ q_phi(s'|s,z') ]
        Uses full states for dynamics (must include coords like x,y).
        """
        # log q_phi(s'|s,z)
        log_p_s_prime = self.dynamics.get_log_prob(state_full, skill, next_state_full)

        # Monte Carlo prior marginal for log p(s'|s) ≈ log (1/L sum q_phi(s'|s,z_i))
        B = state_full.size(0)
        L = num_prior_samples
        z_prior = torch.rand(B, L, self.skill_dim, device=self.device) * 2 - 1

        s_expanded = state_full.unsqueeze(1).expand(-1, L, -1)          # [B, L, D_full]
        sp_expanded = next_state_full.unsqueeze(1).expand(-1, L, -1)    # [B, L, D_full]

        # Flatten for vectorized log_prob
        s_flat = s_expanded.reshape(B * L, -1)
        sp_flat = sp_expanded.reshape(B * L, -1)
        z_flat = z_prior.reshape(B * L, -1)

        log_p_s_prime_prior_flat = self.dynamics.get_log_prob(s_flat, z_flat, sp_flat)  # [B*L]
        log_p_s_prime_prior = log_p_s_prime_prior_flat.view(B, L)                       # [B, L]

        log_avg_p_prior = torch.logsumexp(log_p_s_prime_prior, dim=1) - torch.log(
            torch.tensor(L, dtype=torch.float32, device=self.device)
        )

        intrinsic_reward = log_p_s_prime - log_avg_p_prior
        return intrinsic_reward.unsqueeze(-1)  # [B, 1]

    def update_dynamics(self, state_full, skill, next_state_full):
        """
        Train dynamics q_phi(Δs | s, z) with full states.
        """
        log_prob = self.dynamics.get_log_prob(state_full, skill, next_state_full)
        loss = -log_prob.mean()
        self.dynamics_optimizer.zero_grad()
        loss.backward()
        self.dynamics_optimizer.step()
        return loss.item()

    def update_policy(self, state, skill, action, next_state, done, s_full_t, ns_full_t):
        """
        Policy/critic update with intrinsic reward.
        - state/next_state: FILTERED states (policy/critic inputs)
        - s_full_t/ns_full_t: FULL states (for intrinsic reward via q_phi)
        """
        with torch.no_grad():
            intrinsic_reward = self._calculate_intrinsic_reward(s_full_t, skill, ns_full_t)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, skill)
            target_q1, target_q2 = self.critic_target(next_state, skill, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = intrinsic_reward + (1 - done) * self.config.AGENT_GAMMA * (
                target_q - self.alpha * next_log_prob
            )

        current_q1, current_q2 = self.critic(state, skill, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(state, skill)
        q1_new, q2_new = self.critic(state, skill, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target update
        tau = self.config.AGENT_TARGET_UPDATE_TAU
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), intrinsic_reward.mean().item()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.dynamics.state_dict(), os.path.join(path, "dynamics.pth"))

    def load(self, path):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pth"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pth"), map_location=self.device)
        )
        self.dynamics.load_state_dict(
            torch.load(os.path.join(path, "dynamics.pth"), map_location=self.device)
        )