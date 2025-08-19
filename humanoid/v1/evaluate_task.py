import gymnasium as gym
import numpy as np
import argparse
import time

from config import Config
from agent import DADSAgent
from planner import MPPIPlanner
from utils import filter_state_for_agent

def main(args):
    """
    This script implements Phase 2 of DADS: Zero-Shot Hierarchical Control.
    It loads a pre-trained agent and uses its learned dynamics model for planning
    to navigate to a specified goal coordinate without any new training.
    """
    config = Config()
    config.ENV_NAME = args.env
    
    # Create the environment with "human" render mode to see the agent
    env = gym.make(config.ENV_NAME, render_mode="human")
    
    # Get state and action dimensions
    sample_state_full, _ = env.reset()
    sample_state_filtered = filter_state_for_agent(sample_state_full, config)
    full_state_dim = len(sample_state_full)
    filtered_state_dim = len(sample_state_filtered)
    action_dim = env.action_space.shape[0]

    # Instantiate the agent and the planner
    agent = DADSAgent(
        filtered_state_dim=filtered_state_dim,
        full_state_dim=full_state_dim,
        action_dim=action_dim,
        config=config
    )
    planner = MPPIPlanner(agent, config)

    # --- 2. Load Pre-trained Models ---
    print(f"--- Loading pre-trained models from: {args.model_path} ---")
    try:
        agent.load(args.model_path)
    except FileNotFoundError:
        print(f"ERROR: Models not found at {args.model_path}.")
        print("Please run train.py first to pre-train the skills.")
        return
        
    # Set models to evaluation mode
    agent.actor.eval()
    agent.dynamics.eval()

    # --- 3. Define the Task ---
    goal_pos = np.array([args.goal_x, args.goal_y])
    SUCCESS_THRESHOLD = 1.5  # How close the agent needs to be to the goal
    print(f"--- Starting Task: Navigate to Goal {goal_pos} ---")

    # --- 4. The Hierarchical Control Loop ---
    state_full, _ = env.reset()
    
    for t in range(args.max_steps):
        # Calculate distance to goal
        # We use the first two elements of the full state, which are the (x,y) coordinates
        current_pos = state_full[:2]
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        
        print(f"Step: {t} | Current Position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}] | Distance to Goal: {dist_to_goal:.2f}")

        # Check for success
        if dist_to_goal < SUCCESS_THRESHOLD:
            print(f"\n--- SUCCESS! Reached goal in {t} high-level steps. ---")
            break

        # --- HRL STEP 1: Plan in the high-level skill space ---
        # The planner uses the dynamics model q(s'|s,z) to find the best skill
        skill_z = planner.plan(state_full)
        
        # --- HRL STEP 2: Execute the chosen skill for HZ steps ---
        # The low-level policy π(a|s,z) executes the skill
        for _ in range(config.HZ):
            env.render() # Render the environment to see the agent
            
            # The policy needs the filtered state (without x,y coords)
            state_filtered = filter_state_for_agent(state_full, config)
            
            # Get low-level action from the policy
            action = agent.select_action(state_filtered, skill_z)
            
            # Take a step in the environment
            next_state_full, _, terminated, truncated, _ = env.step(action)
            
            state_full = next_state_full
            
            # If the agent falls over or the episode ends, break the inner loop
            if terminated or truncated:
                break
        
        if terminated or truncated:
            print("\n--- Episode terminated early (agent fell). ---")
            break
    
    # --- 5. Cleanup and Final Status ---
    if dist_to_goal >= SUCCESS_THRESHOLD:
        print(f"\n--- FAILED: Timed out after {args.max_steps} high-level steps. ---")
        print(f"--- Final distance to goal: {dist_to_goal:.2f} ---")

    print("--- Evaluation finished. Closing environment. ---")
    # Keep the window open for a few seconds to see the final state
    time.sleep(5)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DADS pre-trained skills on a navigation task.")
    parser.add_argument("--env", type=str, default="Ant-v5", help="Environment name (e.g., Ant-v5, HalfCheetah-v5, Humanoid-v5)")
    parser.add_argument("--goal_x", type=float, default=10.0, help="Target X coordinate for the goal.")
    parser.add_argument("--goal_y", type=float, default=10.0, help="Target Y coordinate for the goal.")
    parser.add_argument("--model_path", type=str, default="./dads_models", help="Path to the saved models from pre-training.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of high-level planning steps.")
    
    args = parser.parse_args()
    main(args)