import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config

def plot_training_curves(config):
    """Loads training logs from CSV and plots comprehensive metrics."""
    log_file = os.path.join(config.LOG_DATA_PATH, "training_logs.csv")
    if not os.path.exists(log_file):
        print(f"Log file not found at {log_file}. Run training first.")
        return
        
    print("--- Plotting Training Curves from CSV ---")
    df = pd.read_csv(log_file)
    
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'DADS Training Progress on {config.ENV_NAME}', fontsize=16)

    # Row 1: Core DADS metrics
    axs[0, 0].plot(df['step'], df['intrinsic_reward'])
    axs[0, 0].set_title('Intrinsic Reward')
    axs[0, 0].set_ylabel('Reward')
    
    axs[0, 1].plot(df['step'], df['critic_loss'])
    axs[0, 1].set_title('Critic Loss')
    axs[0, 1].set_ylabel('Loss')
    
    axs[0, 2].plot(df['step'], df['actor_loss'])
    axs[0, 2].set_title('Actor Loss')
    axs[0, 2].set_ylabel('Loss')

    # Row 2: Episode and exploration metrics
    axs[1, 0].plot(df['step'], df['episode_return'])
    axs[1, 0].set_title('Episode Return')
    axs[1, 0].set_ylabel('Return')
    
    axs[1, 1].plot(df['step'], df['episode_length'])
    axs[1, 1].set_title('Episode Length')
    axs[1, 1].set_ylabel('Steps')
    
    axs[1, 2].plot(df['step'], df['skill_diversity'])
    axs[1, 2].set_title('Skill Diversity')
    axs[1, 2].set_ylabel('Entropy')

    # Row 3: Advanced metrics
    axs[2, 0].plot(df['step'], df['state_coverage'])
    axs[2, 0].set_title('State Coverage')
    axs[2, 0].set_ylabel('Coverage')
    
    axs[2, 1].plot(df['step'], df['dynamics_prediction_error'])
    axs[2, 1].set_title('Dynamics Prediction Error')
    axs[2, 1].set_ylabel('Relative MSE')
    
    axs[2, 2].plot(df['step'], df['skill_entropy'])
    axs[2, 2].set_title('Skill Usage Entropy')
    axs[2, 2].set_ylabel('Entropy')

    for ax in axs.flat:
        ax.set_xlabel('Training Steps')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_filename = os.path.join(config.PLOT_SAVE_PATH, f"training_progress_{config.ENV_NAME}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved training progress plot to {plot_filename}")
    plt.close()

def plot_trajectories_from_log(config):
    """Loads trajectory data and creates the diversity plot."""
    data_file = os.path.join(config.LOG_DATA_PATH, "skill_trajectories.npz")
    if not os.path.exists(data_file): 
        print(f"Trajectory data not found at {data_file}")
        return
    
    print("--- Plotting Skill Trajectories from Log ---")
    data = np.load(data_file)
    trajectories = [data[arr] for arr in data.files]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], alpha=0.7, linewidth=2)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i], marker='o', s=100, zorder=10, edgecolor='white')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[i], marker='x', s=150, zorder=10)

    plt.title(f'Discovered Skill Trajectories for {config.ENV_NAME}', fontsize=14)
    plt.xlabel('X-Position', fontsize=12)
    plt.ylabel('Y-Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(['Start (o)', 'End (x)'], loc='upper right')
    
    plot_filename = os.path.join(config.PLOT_SAVE_PATH, f"skill_trajectories_{config.ENV_NAME}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved skill trajectory plot to {plot_filename}")
    plt.close()

def plot_variance_from_log(config):
    """Loads variance data and creates the predictability plot."""
    data_file = os.path.join(config.LOG_DATA_PATH, "skill_variance.npy")
    if not os.path.exists(data_file): 
        print(f"Variance data not found at {data_file}")
        return

    print("--- Plotting Skill Variance from Log ---")
    variance_data = np.load(data_file, allow_pickle=True)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(variance_data)))

    for i, skill_rollouts in enumerate(variance_data):
        max_len = max(len(traj) for traj in skill_rollouts)
        padded = np.array([np.pad(t, ((0, max_len - len(t)), (0, 0)), 'edge') for t in skill_rollouts])
        position_std = np.linalg.norm(np.std(padded, axis=0), axis=1)
        plt.plot(range(max_len), position_std, color=colors[i], label=f'Skill {i+1}', linewidth=2)

    plt.title(f'Trajectory Standard Deviation for Fixed Skills ({config.ENV_NAME})', fontsize=14)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Position Std. Dev. (L2 Norm)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = os.path.join(config.PLOT_SAVE_PATH, f"skill_variance_{config.ENV_NAME}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved skill variance plot to {plot_filename}")
    plt.close()

def plot_heatmap_from_log(config):
    """Loads heatmap data and creates the skill space plot."""
    if config.NUM_SKILLS != 2: 
        print("Heatmap requires NUM_SKILLS = 2")
        return
        
    data_file = os.path.join(config.LOG_DATA_PATH, "skill_heatmap.npy")
    if not os.path.exists(data_file): 
        print(f"Heatmap data not found at {data_file}")
        return

    print("--- Plotting Skill Heatmap from Log ---")
    final_x_positions = np.load(data_file)
    grid_size = final_x_positions.shape[0]
    z_vals = np.linspace(-1.0, 1.0, grid_size)

    plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(z_vals, z_vals, final_x_positions, cmap='viridis', shading='auto')
    plt.colorbar(c, label='Final X-Position')
    plt.title(f'Skill Space Heatmap ({config.ENV_NAME})', fontsize=14)
    plt.xlabel('Skill Dimension 1', fontsize=12)
    plt.ylabel('Skill Dimension 2', fontsize=12)
    
    plot_filename = os.path.join(config.PLOT_SAVE_PATH, f"skill_heatmap_{config.ENV_NAME}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved skill heatmap plot to {plot_filename}")
    plt.close()

if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.PLOT_SAVE_PATH):
        os.makedirs(cfg.PLOT_SAVE_PATH)
        
    plot_training_curves(cfg)
    plot_trajectories_from_log(cfg)
    plot_variance_from_log(cfg)
    plot_heatmap_from_log(cfg)
