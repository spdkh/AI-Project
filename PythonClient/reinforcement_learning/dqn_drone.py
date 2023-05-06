import setup_path
import gym
import airgym
import time
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import optuna


# def objective(trial):
#     # Define hyperparameters to tune
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#     training_freq = trial.suggest_categorical("train_freq", [4, 8])
#     buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5)])
#     exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.1, 0.2)
#
#     # Define AirSim environment and RL model with hyperparameters
#     env = DummyVecEnv(
#         [
#             lambda: gym.make(
#                 "airgym:airsim-drone-sample-v0",
#                 ip_address="127.0.0.1",
#                 step_length=0.25,
#                 image_shape=(84, 84, 1),
#             )
#         ]
#     )
#     env = VecTransposeImage(env)
#     model = DQN(
#         "CnnPolicy",
#         env,
#         learning_rate=learning_rate,
#         verbose=1,
#         batch_size=batch_size,
#         train_freq=training_freq,
#         target_update_interval=1000,
#         learning_starts=1000,
#         buffer_size=buffer_size,
#         max_grad_norm=10,
#         exploration_fraction=exploration_fraction,
#         exploration_final_eps=0.01,
#         device="cuda",
#         tensorboard_log="./tb_logs/",
#     )
#
#     # Define evaluation callback and train model for specified number of timesteps
#     eval_callback = EvalCallback(
#         env,
#         callback_on_new_best=None,
#         n_eval_episodes=5,
#         best_model_save_path=".",
#         log_path=".",
#         eval_freq=1000,
#     )
#     model.learn(
#         total_timesteps=5e3,
#         tb_log_name="dqn_airsim_drone_run",
#         callback=[eval_callback],
#     )
#
#     # Evaluate model and return the mean reward
#     mean_reward = eval_callback.best_mean_reward
#     return mean_reward
#
#
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
#
# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value:.3f}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

print('CUDA?', torch.cuda.is_available())
# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.0025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=1000,
    learning_starts=1000,
    buffer_size=50000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=1000,
)
callbacks.append(eval_callback)

kwargs = dict()
kwargs["callback"] = callbacks
# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e4,
    tb_log_name="dqn_airsim_drone_run" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")
