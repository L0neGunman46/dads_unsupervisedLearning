import os
import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def filter_state_for_agent(state, config):
    """Filter state based on environment-specific requirements"""
    if config.EXCLUDE_COORDS_FROM_STATE:
        filtered_state = np.delete(state, config.EXCLUDE_COORDS_FROM_STATE)
        return filtered_state
    return state

def _run_episode(env, agent, skill, config, max_steps=1000):
    """Helper function to run a single episode with a fixed skill."""
    state, _ = env.reset()
    state = filter_state_for_agent(state, config)
    trajectory = []
    full_states = []
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        action = agent.select_action(state, skill)
        next_state, _, terminated, truncated, _ = env.step(action)
        
        # Store full state for trajectory (including coordinates)
        full_states.append(next_state)
        
        # Filter state for agent
        next_state = filter_state_for_agent(next_state, config)
        
        # For trajectory visualization, use x,y coordinates if available
        if len(full_states[-1]) >= 2:
            trajectory.append([full_states[-1][0], full_states[-1][1]])
        else:
            trajectory.append([full_states[-1][0], 0])  # Use x-coordinate only
            
        state = next_state
        done = terminated or truncated
        steps += 1
        
    return np.array(trajectory), np.array(full_states)

def calculate_skill_diversity(skill_usage_counts):
    """Calculate skill diversity using entropy of skill usage"""
    if not skill_usage_counts:
        return 0.0
    
    total_usage = sum(skill_usage_counts.values())
    probabilities = [count / total_usage for count in skill_usage_counts.values()]
    entropy = -sum(p * np.log(p + 1e-8) for p in probabilities)
    
    # Normalize by log of number of skills
    max_entropy = np.log(len(skill_usage_counts) + 1e-8)
    return entropy / (max_entropy + 1e-8)

def calculate_state_coverage(visited_states):
    """Calculate state space coverage using K-nearest neighbors"""
    if len(visited_states) < 10:
        return 0.0
    
    states_array = np.array(visited_states)
    if states_array.shape[1] == 0:
        return 0.0
    
    # Use k-NN to estimate local density
    k = min(10, len(visited_states) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(states_array)
    distances, _ = nbrs.kneighbors(states_array)
    
    # Average distance to k-th nearest neighbor (excluding self)
    avg_distance = np.mean(distances[:, -1])
    
    # Normalize by state space bounds (approximate)
    state_range = np.ptp(states_array, axis=0).mean()
    coverage = 1.0 / (1.0 + avg_distance / (state_range + 1e-8))
    
    return coverage

def evaluate_dynamics_prediction_accuracy(agent, replay_buffer, config, num_samples=100):
    """Evaluate how accurately the dynamics model predicts state transitions"""
    if len(replay_buffer) < num_samples:
        return 0.0
    
    agent.dynamics.eval()
    
    # Sample transitions and get the full states
    _, skills, _, _, _, _, states_full, next_states_full = replay_buffer.sample(num_samples)
    
    states_tensor = torch.FloatTensor(states_full).to(config.DEVICE)
    skills_tensor = torch.FloatTensor(skills).to(config.DEVICE)
    next_states_tensor = torch.FloatTensor(next_states_full).to(config.DEVICE)
    
    with torch.no_grad():
        predicted_next_states = agent.dynamics.predict(states_tensor, skills_tensor)
        
        mse = torch.mean((predicted_next_states - next_states_tensor) ** 2).item()
        state_variance = torch.var(next_states_tensor).item()
        relative_error = mse / (state_variance + 1e-8)
    
    agent.dynamics.train()
    return relative_error

def generate_and_save_skill_trajectories(agent, config):
    """Generate trajectory data for skill diversity analysis and saves it."""
    print("\n--- Generating and Saving Data for Plot 1: Skill Trajectory Diversity ---")
    
    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_EPISODE_STEPS)
    agent.actor.eval()

    all_trajectories = []
    for i in tqdm(range(config.NUM_SKILLS_TO_PLOT), desc="Generating Skill Trajectories"):
        skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)
        trajectory, _ = _run_episode(env, agent, skill, config)
        all_trajectories.append(trajectory)
    
    if not os.path.exists(config.LOG_DATA_PATH):
        os.makedirs(config.LOG_DATA_PATH)
    data_filename = os.path.join(config.LOG_DATA_PATH, "skill_trajectories.npz")
    
    # Save each trajectory as a separate array
    np.savez(data_filename, *all_trajectories)
    
    print(f"Saved trajectory data to {data_filename}")
    env.close()

def generate_and_save_skill_variance(agent, config):
    """Generate trajectory data for skill variance analysis and saves it."""
    print("\n--- Generating and Saving Data for Plot 2: Skill Variance ---")
    
    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_EPISODE_STEPS)
    agent.actor.eval()

    variance_data = []
    for i in tqdm(range(config.NUM_SKILLS_FOR_VARIANCE_PLOT), desc="Generating Skill Variance Data"):
        skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)
        skill_rollouts = []
        for _ in range(config.NUM_ROLLOUTS_PER_SKILL):
            trajectory, _ = _run_episode(env, agent, skill, config)
            skill_rollouts.append(trajectory)
        variance_data.append(skill_rollouts)

    data_filename = os.path.join(config.LOG_DATA_PATH, "skill_variance.npy")
    np.save(data_filename, np.array(variance_data, dtype=object))
    
    print(f"Saved skill variance data to {data_filename}")
    env.close()

def generate_and_save_skill_heatmap(agent, config):
    """Generate grid data for the skill space heatmap and saves it."""
    if config.NUM_SKILLS != 2:
        print("\nSkipping heatmap data generation: requires NUM_SKILLS to be 2.")
        return
        
    print("\n--- Generating and Saving Data for Plot 3: Skill Space Heatmap ---")
    
    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_EPISODE_STEPS)
    agent.actor.eval()

    grid_size = 20
    z1_vals = np.linspace(-1.0, 1.0, grid_size)
    z2_vals = np.linspace(-1.0, 1.0, grid_size)
    final_x_positions = np.zeros((grid_size, grid_size))

    for i, z1 in enumerate(tqdm(z1_vals, desc="Generating Heatmap Data")):
        for j, z2 in enumerate(z2_vals):
            skill = np.array([z1, z2])
            trajectory, full_states = _run_episode(env, agent, skill, config)
            # Use final x-coordinate from full state
            final_x_positions[j, i] = full_states[-1, 0] if len(full_states) > 0 else 0

    data_filename = os.path.join(config.LOG_DATA_PATH, "skill_heatmap.npy")
    np.save(data_filename, final_x_positions)
    print(f"Saved skill heatmap data to {data_filename}")
    env.close()
