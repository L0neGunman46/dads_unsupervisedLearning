import os
import time
import numpy as np
import torch
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

# Swap this import to your chosen config (single or per-env file)
from config import Config
# e.g., from configs.ant import Config

from replay_buffer import ReplayBuffer
from agent import DADSAgent
from utils import (
    generate_and_save_skill_trajectories,
    generate_and_save_skill_variance,
    generate_and_save_skill_heatmap,
    calculate_skill_diversity,
    calculate_state_coverage,
    evaluate_dynamics_prediction_accuracy,
)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(agent, replay_buffer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=agent.device)

        start_step = checkpoint["step"]
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.dynamics_optimizer.load_state_dict(checkpoint["dynamics_optimizer"])

        with open(checkpoint["replay_buffer_path"], "rb") as f:
            replay_buffer.buffer = pickle.load(f)

        print(f"=> Loaded checkpoint '{filename}' (step {start_step})")
        return start_step, agent, replay_buffer
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return 0, agent, replay_buffer


def filter_state_for_agent(state, config):
    if config.EXCLUDE_COORDS_FROM_STATE:
        return np.delete(state, config.EXCLUDE_COORDS_FROM_STATE)
    return state


def main():
    config = Config()

    # Ensure output directories exist
    for path in [
        config.MODEL_SAVE_PATH,
        config.CHECKPOINT_PATH,
        config.LOG_DATA_PATH,
        config.PLOT_SAVE_PATH,
    ]:
        os.makedirs(path, exist_ok=True)

    # Environment
    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_EPISODE_STEPS)

    # Determine both filtered and full state dimensions
    sample_state_full, _ = env.reset()
    sample_state_filtered = filter_state_for_agent(sample_state_full, config)
    full_state_dim = len(sample_state_full)
    filtered_state_dim = len(sample_state_filtered)
    action_dim = env.action_space.shape[0]

    # Agent
    agent = DADSAgent(
        filtered_state_dim=filtered_state_dim,
        full_state_dim=full_state_dim,
        action_dim=action_dim,
        config=config,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    # Try to load from checkpoint
    start_step, agent, replay_buffer = load_checkpoint(
        agent, replay_buffer, os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth.tar")
    )

    # Metrics accumulators (recent averages)
    temp_rewards, temp_c_loss, temp_a_loss, temp_d_loss = [], [], [], []
    temp_episode_returns, temp_episode_lengths = [], []
    skill_usage_counts = {}
    visited_states = deque(maxlen=10000)

    print(f"--- Starting DADS Pre-training on {config.DEVICE} for {config.ENV_NAME} ---")

    # Initialize episode
    state_full, _ = env.reset()
    state = filter_state_for_agent(state_full, config)
    prev_state_full = state_full.copy()
    current_skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)

    episode_return = 0.0
    episode_length = 0

    start_time = time.time()

    for step in range(start_step + 1, config.NUM_PRETRAIN_STEPS + 1):
        # Action selection
        if len(replay_buffer) < config.INITIAL_COLLECT_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, current_skill)

        # Step environment
        next_state_full, reward, terminated, truncated, _ = env.step(action)
        next_state = filter_state_for_agent(next_state_full, config)
        done = terminated or truncated

        # Store transition (both filtered and full states)
        replay_buffer.push(
            state,
            current_skill,
            action,
            reward,
            next_state,
            done,
            state_full=prev_state_full,
            next_state_full=next_state_full.copy(),
        )
        print("Replay buffer push done")

        # Update episode stats
        episode_return += reward
        episode_length += 1
        # Track 2D position coverage if available
        if len(state) >= 2:
            visited_states.append(state[:2])
        else:
            visited_states.append(state[:1])

        # Track skill usage
        skill_key = tuple(np.round(current_skill, 2))
        skill_usage_counts[skill_key] = skill_usage_counts.get(skill_key, 0) + 1

        # Advance state
        state = next_state
        prev_state_full = next_state_full.copy()

        # If episode done: reset environment and pick new skill
        if done:
            temp_episode_returns.append(episode_return)
            temp_episode_lengths.append(episode_length)

            state_full, _ = env.reset()
            state = filter_state_for_agent(state_full, config)
            prev_state_full = state_full.copy()
            current_skill = np.random.uniform(-1.0, 1.0, size=config.NUM_SKILLS)
            episode_return = 0.0
            episode_length = 0

        # Training updates once buffer has enough data
        print("Training updates startted")
        if len(replay_buffer) > config.INITIAL_COLLECT_STEPS:
            (
                s_b,
                z_b,
                a_b,
                _,
                ns_b,
                d_b,
                s_full_b,
                ns_full_b,
            ) = replay_buffer.sample(config.AGENT_BATCH_SIZE)

            # Tensors for policy/critic (filtered)
            s_t = torch.as_tensor(s_b, dtype=torch.float32, device=config.DEVICE)
            z_t = torch.as_tensor(z_b, dtype=torch.float32, device=config.DEVICE)
            a_t = torch.as_tensor(a_b, dtype=torch.float32, device=config.DEVICE)
            ns_t = torch.as_tensor(ns_b, dtype=torch.float32, device=config.DEVICE)
            d_t = torch.as_tensor(d_b, dtype=torch.float32, device=config.DEVICE)

            # Tensors for dynamics (full)
            s_full_t = torch.as_tensor(s_full_b, dtype=torch.float32, device=config.DEVICE)
            ns_full_t = torch.as_tensor(ns_full_b, dtype=torch.float32, device=config.DEVICE)

            # Update dynamics
            dynamics_loss = agent.update_dynamics(s_full_t, z_t, ns_full_t)

            # Update policy/critic with intrinsic reward
            critic_loss, actor_loss, intrinsic_reward = agent.update_policy(
                s_t, z_t, a_t, ns_t, d_t, s_full_t=s_full_t, ns_full_t=ns_full_t
            )

            temp_d_loss.append(dynamics_loss)
            temp_c_loss.append(critic_loss)
            temp_a_loss.append(actor_loss)
            temp_rewards.append(intrinsic_reward)

        # Lightweight terminal status
        print("Step", step)
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

            elapsed_time = time.time() - start_time
            steps_per_sec = (step - start_step) / elapsed_time if elapsed_time > 0 else 0

            print(
                f"Step: {step}/{config.NUM_PRETRAIN_STEPS} | "
                f"Intrinsic: {avg_reward:.3f} | Return: {avg_episode_return:.1f} | "
                f"Diversity: {skill_diversity:.3f} | Coverage: {state_coverage:.3f} | "
                f"DynErr: {dynamics_error:.3f} | it/s: {steps_per_sec:.1f}"
            )

            # Reset short-term accumulators
            temp_rewards, temp_c_loss, temp_a_loss, temp_d_loss = [], [], [], []
            temp_episode_returns, temp_episode_lengths = [], []

        # Paper-style plots at intervals
        if (
            step % getattr(config, "PLOT_INTERVAL", 10000) == 0
            and len(replay_buffer) > config.INITIAL_COLLECT_STEPS
        ):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1) Skill trajectories (diversity)
            generate_and_save_skill_trajectories(agent, config)

            # 2) Skill variance (predictability)
            generate_and_save_skill_variance(agent, config)

            # 3) Heatmap if 2D skill space
            generate_and_save_skill_heatmap(agent, config)

            # 4) Compact training snapshot panel
            # recompute recent aggregates to plot
            avg_reward = np.mean(temp_rewards) if temp_rewards else 0
            avg_c_loss = np.mean(temp_c_loss) if temp_c_loss else 0
            avg_a_loss = np.mean(temp_a_loss) if temp_a_loss else 0
            avg_d_loss = np.mean(temp_d_loss) if temp_d_loss else 0
            skill_diversity = calculate_skill_diversity(skill_usage_counts)
            state_coverage = calculate_state_coverage(list(visited_states))
            dynamics_error = 0
            if len(replay_buffer) > 1000:
                dynamics_error = evaluate_dynamics_prediction_accuracy(agent, replay_buffer, config)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Training Snapshot step {step} ({config.ENV_NAME})", fontsize=14)

            axs[0, 0].bar(["Intrinsic"], [avg_reward])
            axs[0, 0].set_title("Intrinsic Reward (recent avg)")
            axs[0, 0].grid(alpha=0.3)

            axs[0, 1].bar(["Critic", "Actor", "Dynamics"], [avg_c_loss, avg_a_loss, avg_d_loss])
            axs[0, 1].set_title("Losses (recent avg)")
            axs[0, 1].grid(alpha=0.3)

            axs[1, 0].bar(["Diversity", "Coverage"], [skill_diversity, state_coverage])
            axs[1, 0].set_ylim(0, 1.0)
            axs[1, 0].set_title("Exploration Metrics")
            axs[1, 0].grid(alpha=0.3)

            axs[1, 1].bar(["Dyn Rel Err"], [dynamics_error])
            axs[1, 1].set_title("Dynamics Prediction Error")
            axs[1, 1].grid(alpha=0.3)

            snap_path = os.path.join(
                config.PLOT_SAVE_PATH, f"training_snapshot_step{step}_{ts}.png"
            )
            plt.tight_layout()
            plt.savefig(snap_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved snapshot plot: {snap_path}")

        # Save checkpoint periodically
        if step % config.CHECKPOINT_FREQUENCY == 0:
            buffer_path = os.path.join(config.CHECKPOINT_PATH, "replay_buffer.pkl")
            with open(buffer_path, "wb") as f:
                pickle.dump(replay_buffer.buffer, f)
            checkpoint_state = {
                "step": step,
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "dynamics_state_dict": agent.dynamics.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic_optimizer": agent.critic_optimizer.state_dict(),
                "dynamics_optimizer": agent.dynamics_optimizer.state_dict(),
                "replay_buffer_path": buffer_path,
            }
            save_checkpoint(
                checkpoint_state,
                filename=os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth.tar"),
            )

    # Save final models
    agent.save(config.MODEL_SAVE_PATH)
    print(f"\n--- Pre-training complete. Final models saved to {config.MODEL_SAVE_PATH} ---")

    # Final visualization pass
    print("\n--- Generating Data for Final Visualizations ---")
    agent.load(config.MODEL_SAVE_PATH)
    generate_and_save_skill_trajectories(agent, config)
    generate_and_save_skill_variance(agent, config)
    generate_and_save_skill_heatmap(agent, config)
    print("\n--- All data generation complete. ---")


if __name__ == "__main__":
    main()