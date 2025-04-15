# -*- coding: utf-8 -*-
"""Main training script for PPO agent on the Maze environment."""

import argparse
import csv
import os
import time

os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure Keras uses TensorFlow backend

from pathlib import Path

import numpy as np
import tensorflow as tf
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from tqdm import tqdm

from maze.envs import Maze
from reinforce.ppo.agent import PPOAgent
from reinforce.ppo.buffer import Buffer

# --- Configuration ---
# Default hyperparameters (can be overridden by command-line arguments)
DEFAULT_CONFIG = {
    "epochs": 50,
    "steps_per_epoch": 4000,
    "save_freq": 10,  # Save weights every N epochs
    "render": False,  # Render environment during training
    "seed": 42,
    "save_dir": "checkpoints",
    # PPO Agent Hyperparameters (can be nested or passed directly)
    "gamma": 0.99,
    "lam": 0.97,  # GAE lambda
    "clip_ratio": 0.2,
    "policy_learning_rate": 3e-4,
    "value_function_learning_rate": 1e-3,
    "train_policy_iterations": 80,
    "train_value_iterations": 80,
    "target_kl": 0.01,
    # Network Parameters (optional, passed to build_actor_critic)
    "conv_filters": 32,
    "conv_kernel_size": 3,
    "dense_units": 128,
}


def train(config):
    """
    Trains the PPO agent on the specified environment.

    Args:
        config (dict): Dictionary containing training configuration and hyperparameters.
    """
    # --- Initialization ---
    print("Initializing training...")
    print(f"Configuration: {config}")

    # --- Setup Logging ---
    # ##: Create save directory if it doesn't exist.
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ##: Setup CSV logging.
    log_filepath = save_dir / "training_log.csv"
    log_file = open(log_filepath, "a", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["Epoch", "Mean Return", "Mean Length", "Num Episodes", "Epoch Duration (s)", "Total Duration (s)"]
    )
    print(f"Logging training metrics to: {log_filepath}")

    # ##:Set random seeds for reproducibility.
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    # ##: Initialize Environment.
    print("Creating environment")
    env = Maze()
    env = ImgObsWrapper(RGBImgObsWrapper(env))

    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"Observation Shape: {observation_shape}, Number of Actions: {num_actions}")

    # ##: Initialize Agent.
    agent_hyperparams = {
        k: v
        for k, v in config.items()
        if k
        in [
            "gamma",
            "lam",
            "clip_ratio",
            "policy_learning_rate",
            "value_function_learning_rate",
            "train_policy_iterations",
            "train_value_iterations",
            "target_kl",
            "seed",
        ]
    }
    network_params = {k: v for k, v in config.items() if k in ["conv_filters", "conv_kernel_size", "dense_units"]}
    agent = PPOAgent(
        env.observation_space, env.action_space, hyperparams=agent_hyperparams, network_params=network_params
    )

    # ##: Initialize Buffer.
    buffer = Buffer(observation_shape, config["steps_per_epoch"], gamma=config["gamma"], lam=config["lam"])

    # --- Training Loop ---
    print("Starting training loop...")
    start_time = time.time()

    observation, _ = env.reset(seed=config["seed"])
    episode_return, episode_length = 0, 0
    total_steps = config["epochs"] * config["steps_per_epoch"]

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        pbar = tqdm(range(config["steps_per_epoch"]), desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for t in pbar:
            if config["render"]:
                env.render()

            logits, action_tensor = agent.sample_action(observation)
            action = action_tensor[0].numpy()

            # ##: Environment step.
            observation_new, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            # ##: Get value estimate and log probability.
            obs_tensor = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), 0)
            value_t = agent.critic(obs_tensor)
            logprobability_t = agent._logprobabilities(logits, action_tensor)

            # ##: Store experience in buffer.
            buffer.store(observation, action, reward, value_t[0].numpy(), logprobability_t[0].numpy())

            # ##: Update observation.
            observation = observation_new

            # ##: Handle episode end or epoch end.
            terminal = done
            if terminal or (t == config["steps_per_epoch"] - 1):
                if not terminal:
                    print(f"\nEpoch {epoch+1} ended mid-episode. Bootstrapping value.")
                    obs_tensor = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), 0)
                    last_value = agent.critic(obs_tensor)[0].numpy()
                else:
                    last_value = 0
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1

                    observation, _ = env.reset()
                    episode_return, episode_length = 0, 0

                buffer.finish_trajectory(last_value)

        # ##: Perform PPO update.
        agent.train(buffer)

        # ##: Log metrics for the epoch.
        epoch_duration = time.time() - epoch_start_time
        mean_return = sum_return / num_episodes if num_episodes > 0 else 0
        mean_length = sum_length / num_episodes if num_episodes > 0 else 0
        print(f"\nEpoch: {epoch + 1}/{config['epochs']} | Duration: {epoch_duration:.2f}s")
        print(f"Mean Return: {mean_return:.2f} | Mean Length: {mean_length:.2f} | Episodes: {num_episodes}")

        # ##: Log metrics to CSV.
        current_total_duration = time.time() - start_time
        log_writer.writerow(
            [epoch + 1, mean_return, mean_length, num_episodes, epoch_duration, current_total_duration]
        )
        log_file.flush()

        # ##: Save agent weights periodically.
        if (epoch + 1) % config["save_freq"] == 0 or (epoch + 1) == config["epochs"]:
            save_path_prefix = os.path.join(config["save_dir"], f"ppo_maze_epoch_{epoch + 1}")
            agent.save_weights(save_path_prefix)

    # --- End of Training ---
    total_duration = time.time() - start_time
    print("\nTraining finished.")
    print(f"Total duration: {total_duration:.2f}s")
    log_file.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on MiniGrid Maze")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument(
        "--steps-per-epoch", type=int, default=DEFAULT_CONFIG["steps_per_epoch"], help="Number of steps per epoch"
    )
    parser.add_argument(
        "--save-freq", type=int, default=DEFAULT_CONFIG["save_freq"], help="Frequency (in epochs) to save weights"
    )
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"], help="Directory to save checkpoints"
    )
    # Add other hyperparameters as needed
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--lam", type=float, default=DEFAULT_CONFIG["lam"])
    parser.add_argument("--clip-ratio", type=float, default=DEFAULT_CONFIG["clip_ratio"])
    parser.add_argument(
        "--plr", type=float, default=DEFAULT_CONFIG["policy_learning_rate"], dest="policy_learning_rate"
    )
    parser.add_argument(
        "--vlr",
        type=float,
        default=DEFAULT_CONFIG["value_function_learning_rate"],
        dest="value_function_learning_rate",
    )
    parser.add_argument(
        "--pi-iters", type=int, default=DEFAULT_CONFIG["train_policy_iterations"], dest="train_policy_iterations"
    )
    parser.add_argument(
        "--v-iters", type=int, default=DEFAULT_CONFIG["train_value_iterations"], dest="train_value_iterations"
    )
    parser.add_argument("--target-kl", type=float, default=DEFAULT_CONFIG["target_kl"])
    parser.add_argument("--conv-filters", type=int, default=DEFAULT_CONFIG["conv_filters"])
    parser.add_argument("--dense-units", type=int, default=DEFAULT_CONFIG["dense_units"])

    args = parser.parse_args()
    config = vars(args)  # Convert argparse Namespace to dict

    train(config)
