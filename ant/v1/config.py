import torch

class Config:
    # Environment
    ENV_NAME = "Ant-v5"

    # Unsupervised DADS pre-training
    NUM_PRETRAIN_STEPS = 1_000_000
    REPLAY_BUFFER_CAPACITY = 1_000_000
    INITIAL_COLLECT_STEPS = 5_000

    # Skill space (continuous)
    # 2D skill space often pairs well with x-y displacement analysis in the paper
    NUM_SKILLS = 2
    SKILL_TYPE = "cont_uniform"

    # SAC (low-level policy)
    AGENT_BATCH_SIZE = 256
    AGENT_LR = 3e-4
    AGENT_GAMMA = 0.99
    AGENT_ALPHA = 0.1
    AGENT_TARGET_UPDATE_TAU = 0.005

    # Skill dynamics (Mixture-of-Gaussians)
    DYNAMICS_BATCH_SIZE = 256
    DYNAMICS_LR = 3e-4
    DYNAMICS_NUM_EXPERTS = 4
    DYNAMICS_LEARN_STD = False

    # Network sizes
    @property
    def HIDDEN_DIM(self):
        return 512

    # Episode length
    @property
    def MAX_EPISODE_STEPS(self):
        return 1000

    # Latent-space MPC (Phase 2)
    HP = 3
    HZ = 10
    MPPI_R = 5
    MPPI_K = 64
    MPPI_GAMMA = 10.0
    MPPI_COV = 0.3

    # Visualization & plotting
    NUM_SKILLS_TO_PLOT = 10
    NUM_SKILLS_FOR_VARIANCE_PLOT = 4
    NUM_ROLLOUTS_PER_SKILL = 10

    # Logging & eval
    EVAL_FREQUENCY = 5_000
    SKILL_DIVERSITY_EVAL_FREQUENCY = 10_000
    LOG_FREQUENCY = 100

    # Device & paths
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "./dads_models"
    PLOT_SAVE_PATH = "./dads_plots"
    LOG_DATA_PATH = "./dads_logs"
    CHECKPOINT_PATH = "./dads_checkpoints"
    CHECKPOINT_FREQUENCY = 25_000

    # Exclude global coords (x, y)
    @property
    def EXCLUDE_COORDS_FROM_STATE(self):
        return [0, 1]