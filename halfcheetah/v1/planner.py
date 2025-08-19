import torch

class MPPIPlanner:
    def __init__(self, agent, config):
        self.dynamics = agent.dynamics
        self.config = config
        self.device = config.DEVICE

    def plan(self, current_state):
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        
        candidate_skills = torch.rand(
            self.config.NUM_CANDIDATE_SEQUENCES, self.config.NUM_SKILLS, device=self.device
        ) * 2 - 1

        state_batch = state_tensor.unsqueeze(0).expand(self.config.NUM_CANDIDATE_SEQUENCES, -1)
        total_rewards = torch.zeros(self.config.NUM_CANDIDATE_SEQUENCES, device=self.device)

        predicted_state = state_batch
        for _ in range(self.config.PLANNING_HORIZON):
            with torch.no_grad():
                next_predicted_state = self.dynamics.predict(predicted_state, candidate_skills)
            
            # Hopper reward: forward_vel + healthy_reward - ctrl_cost
            # We only model the forward velocity part for planning.
            # dt=0.008 for Hopper-v5
            forward_velocity = (next_predicted_state[:, 0] - predicted_state[:, 0]) / 0.008 
            total_rewards += forward_velocity
            predicted_state = next_predicted_state

        exp_rewards = torch.exp(self.config.MPPI_GAMMA * (total_rewards - torch.max(total_rewards)))
        weights = exp_rewards / (exp_rewards.sum() + 1e-8)
        
        best_skill = torch.sum(weights.unsqueeze(-1) * candidate_skills, dim=0)
        return best_skill.detach().cpu().numpy()


    @torch.no_grad()
    def _simulate_batch(self, s0, Z_seq):
        """
        s0: [B, D_state] (full state dimension)
        Z_seq: [B, HP, skill_dim]
        Returns cumulative reward per batch element based on predicted movement.
        """
        B = s0.shape[0]
        state = s0.clone()
        total_rewards = torch.zeros(B, device=self.device)

        # dt_env = 0.008 # MuJoCo simulation timestep, used for velocities if needed
        
        for i in range(self.HP):
            z_i = Z_seq[:, i, :]  # [B, skill_dim]
            for _ in range(self.HZ):
                next_state = self.dynamics.predict(state, z_i)
                
                # --- START OF CHANGE: More Generic Locomotion Reward ---
                # This reward is a proxy for general movement/exploration during planning.
                # For specific downstream tasks, this would be the actual task reward.
                movement_reward = torch.zeros_like(total_rewards)
                
                # Check if state has at least 2 dimensions (x, y) for L2 norm
                if next_state.shape[-1] >= 2 and state.shape[-1] >= 2:
                    # Calculate L2 norm of (x,y) position change
                    delta_pos_xy = next_state[:, :2] - state[:, :2]
                    movement_reward = torch.linalg.norm(delta_pos_xy, dim=-1)
                elif next_state.shape[-1] >= 1 and state.shape[-1] >= 1:
                    # If only x is available (e.g., Hopper, HalfCheetah when focusing on x)
                    # Use absolute change in x-position
                    delta_pos_x = next_state[:, 0] - state[:, 0]
                    movement_reward = torch.abs(delta_pos_x)
                
                total_rewards += movement_reward
                state = next_state
                # --- END OF CHANGE ---

        return total_rewards