import torch

class Config:
    ENV_NAME = "Ant-v5" 

    # DADS Pre-training
    NUM_PRETRAIN_STEPS = 1_000_000 # Often higher for Humanoid (e.g., 2M-5M)
    REPLAY_BUFFER_CAPACITY = 1_000_000
    INITIAL_COLLECT_STEPS = 5000
    
    # Skill Discovery
    NUM_SKILLS = 8 # A good starting point (paper suggests up to 128 for cont. skills)
    SKILL_TYPE = "cont_uniform" # Keep as continuous uniform for DADS

    # SAC Agent (Low-level policy)
    AGENT_BATCH_SIZE = 256
    AGENT_LR = 3e-4
    AGENT_GAMMA = 0.99
    AGENT_ALPHA = 0.1 # Consider adding auto-tuning for this (more advanced)
    AGENT_TARGET_UPDATE_TAU = 0.005
    
    # Skill Dynamics Model
    DYNAMICS_BATCH_SIZE = 256
    DYNAMICS_LR = 3e-4
    DYNAMICS_NUM_EXPERTS = 4 # From paper for MoG
    DYNAMICS_LEARN_STD = False # From paper (fixed std per component)

    # Network Architecture - Environment specific properties
    @property
    def HIDDEN_DIM(self):
        env_dims = {
            "Hopper-v5": 256,
            "HalfCheetah-v5": 256,  # Smaller hidden dims for simpler dynamics
            "Ant-v5": 512,
            "Humanoid-v5": 1024, # Larger hidden dims for complex dynamics
        }
        return env_dims.get(self.ENV_NAME, 256) # Default for other environments

    # Environment-specific episode lengths
    @property
    def MAX_EPISODE_STEPS(self):
        env_steps = {
            "HalfCheetah-v5": 1000,
            "Ant-v5": 1000,
            "Humanoid-v5": 1000,
        }
        return env_steps.get(self.ENV_NAME, 1000) # Default for other environments
    
    # Planning & Evaluation (for latent-space MPC - test time / visualization)
    HP = 3          # Plan length in primitives
    HZ = 10         # Primitive hold duration (how many low-level steps per primitive)
    MPPI_R = 5      # Refinement steps per replan
    MPPI_K = 64     # Samples per refinement
    MPPI_GAMMA = 10.0 # MPPI temperature
    MPPI_COV = 0.3  # Std for sampling skills in MPPI

    # Visualization & Plotting
    NUM_SKILLS_TO_PLOT = 10
    NUM_SKILLS_FOR_VARIANCE_PLOT = 4
    NUM_ROLLOUTS_PER_SKILL = 10
    
    # Enhanced Metrics
    EVAL_FREQUENCY = 5000
    SKILL_DIVERSITY_EVAL_FREQUENCY = 10000
    
    # Device & Paths
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "./dads_models"
    PLOT_SAVE_PATH = "./dads_plots"
    LOG_DATA_PATH = "./dads_logs"
    LOG_FREQUENCY = 1 # Set to 100 or 1000 for faster training
    
    # Checkpointing
    CHECKPOINT_PATH = "./dads_checkpoints"
    CHECKPOINT_FREQUENCY = 25000
    
    # Environment-specific state filtering for policy/critic inputs
    @property
    def EXCLUDE_COORDS_FROM_STATE(self):
        exclude_coords = {
            "Hopper-v5": [0],         # Exclude x-coordinate (position along the ground)
            "HalfCheetah-v5": [0],    # Exclude x-coordinate
            "Ant-v5": [0, 1],         # Exclude x,y coordinates
            "Humanoid-v5": [0, 1]     # Exclude x,y coordinates
        }
        return exclude_coords.get(self.ENV_NAME, [])