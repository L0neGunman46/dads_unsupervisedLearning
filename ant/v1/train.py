import os
import time
import numpy as np
import torch
import gymnasium as gym
import pickle
import csv
from collections import deque

from config import Config
from replay_buffer import ReplayBuffer
from agent import DADSAgent
from planner import MPPIPlanner
from utils import (
    generate_and_save_skill_trajectories, 
    generate_and_save_skill_variance, 
    generate_and_save_skill_heatmap,
    calculate_skill_diversity,
    calculate_state_coverage,
    evaluate_dynamics_prediction_accuracy
)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the training state to a file."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(agent, replay_buffer, filename="checkpoint.pth.tar"):
    """Loads the training state from a file."""
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        
        start_step = checkpoint['step']
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.dynamics_optimizer.load_state_dict(checkpoint['dynamics_optimizer'])
        
        with open(checkpoint['replay_buffer_path'], 'rb') as f:
            replay_buffer.buffer = pickle.load(f)
            
        print(f"=> Loaded checkpoint '{filename}' (step {start_step})")
        return start_step, agent, replay_buffer
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return 0, agent, replay_buffer

def filter_state_for_agent(state, config):
    """Filter state based on environment-specific requirements"""
    if config.EXCLUDE_COORDS_FROM_STATE:
        filtered_state = np.delete(state, config.EXCLUDE_COORDS_FROM_STATE)
        return filtered_state
    return state

def main():
    config = Config()
    
    for path in [config.MODEL_SAVE_PATH, config.CHECKPOINT_PATH, config.LOG_DATA_PATH, config.PLOT_SAVE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
        
    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_EPISODE_STEPS)
    
    # --- START OF MAJOR CHANGES (Initialization) ---

    # Get BOTH full and filtered state dimensions
    sample_state_full, _ = env.reset()
    sample_state_filtered = filter_state_for_agent(sample_state_full, config)

    full_state_dim = len(sample_state_full)
    filtered_state_dim = len(sample_state_filtered)
    action_dim = env.action_space.shape[0]

    # Pass both dimensions to the agent constructor
    agent = DADSAgent(
        filtered_state_dim=filtered_state_dim,
        full_state_dim=full_state_dim,
        action_dim=action_dim,
        config=config
    )
    
    # --- END OF MAJOR CHANGES (Initialization) ---

    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    start_step, agent, replay_buffer = load_checkpoint(
        agent, replay_buffer, os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth.tar")
    )

    # Enhanced logging setup
    log_file_path = os.path.join(config.LOG_DATA_PATH, "training_logs.csv")
    log_file_exists = os.path.isfile(log_file_path) and start_step > 0
    log_file = open(log_file_path, 'a' if log_file_exists else 'w', newline='')
    log_writer = csv.writer(log_file)
    
    if not log_file_exists:
        log_writer.writerow([
            'step', 'intrinsic_reward', 'critic_loss', 'actor_loss', 'dynamics_loss',
            'episode_return', 'episode_length', 'skill_diversity', 'state_coverage',
            'dynamics_prediction_error', 'exploration_bonus', 'skill_entropy'
        ])

    # Metrics tracking
    temp_rewards, temp_c_loss, temp_a_loss, temp_d_loss = [], [], [], []
    temp_episode_returns, temp_episode_lengths = [], []
    skill_usage_counts = {}
    visited_states = deque(maxlen=10000)
    
    print(f"--- Starting DADS Pre-training on {config.DEVICE} for {config.ENV_NAME} ---")
    
    # --- START OF CHANGES (State Tracking) ---
    state_full, _ = env.reset()
    state = filter_state_for_agent(state_full, config)
    prev_state_full = state_full.copy()
    # --- END OF CHANGES (State Tracking) ---
    
    current_skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)
    
    episode_return = 0
    episode_length = 0
    
    start_time = time.time()
    for step in range(start_step + 1, config.NUM_PRETRAIN_STEPS + 1):
        if len(replay_buffer) < config.INITIAL_COLLECT_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, current_skill)
            
        # --- START OF CHANGES (Step and Buffer Push) ---
        next_state_full, reward, terminated, truncated, _ = env.step(action)
        next_state = filter_state_for_agent(next_state_full, config)
        done = terminated or truncated
        
        replay_buffer.push(
            state, current_skill, action, reward, next_state, done,
            state_full=prev_state_full, next_state_full=next_state_full.copy()
        )
        # --- END OF CHANGES (Step and Buffer Push) ---
        
        episode_return += reward
        episode_length += 1
        visited_states.append(state[:2] if len(state) >= 2 else state[:1])
        
        skill_key = tuple(np.round(current_skill, 2))
        skill_usage_counts[skill_key] = skill_usage_counts.get(skill_key, 0) + 1
        
        # --- START OF CHANGES (State Advancement) ---
        state = next_state
        prev_state_full = next_state_full.copy()
        # --- END OF CHANGES (State Advancement) ---

        if done:
            temp_episode_returns.append(episode_return)
            temp_episode_lengths.append(episode_length)
            
            # --- START OF CHANGES (Reset Logic) ---
            state_full, _ = env.reset()
            state = filter_state_for_agent(state_full, config)
            prev_state_full = state_full.copy()
            # --- END OF CHANGES (Reset Logic) ---

            current_skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)
            episode_return = 0
            episode_length = 0

        # --- START OF MAJOR CHANGES (Training Update) ---
        if len(replay_buffer) > config.INITIAL_COLLECT_STEPS:
            # Unpack all 8 items from the replay buffer
            (s_b, z_b, a_b, _, ns_b, d_b, s_full_b, ns_full_b) = replay_buffer.sample(config.AGENT_BATCH_SIZE)

            # Tensors for policy/critic (filtered states)
            s_t = torch.FloatTensor(s_b).to(config.DEVICE)
            z_t = torch.FloatTensor(z_b).to(config.DEVICE)
            a_t = torch.FloatTensor(a_b).to(config.DEVICE)
            ns_t = torch.FloatTensor(ns_b).to(config.DEVICE)
            d_t = torch.FloatTensor(d_b).to(config.DEVICE)

            # Tensors for dynamics model (full states)
            s_full_t = torch.FloatTensor(s_full_b).to(config.DEVICE)
            ns_full_t = torch.FloatTensor(ns_full_b).to(config.DEVICE)

            # Pass the correct states to the correct functions
            dynamics_loss = agent.update_dynamics(s_full_t, z_t, ns_full_t)
            critic_loss, actor_loss, intrinsic_reward = agent.update_policy(
                s_t, z_t, a_t, ns_t, d_t, s_full_t=s_full_t, ns_full_t=ns_full_t
            )
            
            temp_d_loss.append(dynamics_loss)
            temp_c_loss.append(critic_loss)
            temp_a_loss.append(actor_loss)
            temp_rewards.append(intrinsic_reward)
        # --- END OF MAJOR CHANGES (Training Update) ---

        if step % config.LOG_FREQUENCY == 0 and len(replay_buffer) > config.INITIAL_COLLECT_STEPS:
            avg_reward = np.mean(temp_rewards) if temp_rewards else 0
            avg_c_loss = np.mean(temp_c_loss) if temp_c_loss else 0
            avg_a_loss = np.mean(temp_a_loss) if temp_a_loss else 0
            avg_d_loss = np.mean(temp_d_loss) if temp_d_loss else 0
            avg_episode_return = np.mean(temp_episode_returns) if temp_episode_returns else 0
            avg_episode_length = np.mean(temp_episode_lengths) if temp_episode_lengths else 0
            
            skill_diversity = calculate_skill_diversity(skill_usage_counts)
            state_coverage = calculate_state_coverage(list(visited_states))
            
            dynamics_error = 0
            if len(replay_buffer) > 1000:
                dynamics_error = evaluate_dynamics_prediction_accuracy(agent, replay_buffer, config)
            
            exploration_bonus = avg_reward
            skill_entropy = -sum(p * np.log(p + 1e-8) for p in 
                                np.array(list(skill_usage_counts.values())) / sum(skill_usage_counts.values())
                                if p > 0) if skill_usage_counts else 0
            
            log_writer.writerow([
                step, avg_reward, avg_c_loss, avg_a_loss, avg_d_loss,
                avg_episode_return, avg_episode_length, skill_diversity, state_coverage,
                dynamics_error, exploration_bonus, skill_entropy
            ])
            log_file.flush()
            
            temp_rewards, temp_c_loss, temp_a_loss, temp_d_loss = [], [], [], []
            temp_episode_returns, temp_episode_lengths = [], []
            
            elapsed_time = time.time() - start_time
            steps_per_sec = (step - start_step) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Step: {step}/{config.NUM_PRETRAIN_STEPS} | "
                  f"Intrinsic Reward: {avg_reward:.3f} | "
                  f"Episode Return: {avg_episode_return:.1f} | "
                  f"Skill Diversity: {skill_diversity:.3f} | "
                  f"State Coverage: {state_coverage:.3f} | "
                  f"Steps/sec: {steps_per_sec:.1f}")

        if step % config.CHECKPOINT_FREQUENCY == 0:
            buffer_path = os.path.join(config.CHECKPOINT_PATH, "replay_buffer.pkl")
            with open(buffer_path, 'wb') as f:
                pickle.dump(replay_buffer.buffer, f)
            checkpoint_state = {
                'step': step,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'dynamics_state_dict': agent.dynamics.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'dynamics_optimizer': agent.dynamics_optimizer.state_dict(),
                'replay_buffer_path': buffer_path,
            }
            save_checkpoint(checkpoint_state, filename=os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth.tar"))

    log_file.close()
    agent.save(config.MODEL_SAVE_PATH)
    print(f"\n--- Pre-training complete. Final models saved to {config.MODEL_SAVE_PATH} ---")

    print("\n--- Generating Data for Final Visualizations ---")
    agent.load(config.MODEL_SAVE_PATH)
    
    generate_and_save_skill_trajectories(agent, config)
    generate_and_save_skill_variance(agent, config)
    generate_and_save_skill_heatmap(agent, config)

    print("\n--- All data generation complete. You can now run plot_from_logs.py ---")
    
if __name__ == "__main__":
    main()
