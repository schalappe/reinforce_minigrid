# -*- coding: utf-8 -*-
"""Script to evaluate a trained PPO agent on the Maze environment."""

import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure Keras uses TensorFlow backend

from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from tqdm import tqdm

from maze.envs import Maze
from reinforce.ppo.agent import PPOAgent

# --- Configuration ---
DEFAULT_EVAL_CONFIG = {
    "env_id": "Maze-v0",
    "num_episodes": 10,  # Number of episodes to evaluate
    "render": True,  # Render environment during evaluation
    "seed": 123,  # Use a different seed for evaluation than training
    "load_dir": "checkpoints",  # Directory containing saved weights
    "weights_prefix": "ppo_maze_epoch_50",  # Specific weights file prefix (e.g., from last epoch)
    "max_episode_steps": 200,  # Maximum steps per evaluation episode
    "save_gif": False,  # Whether to save a GIF of the evaluation episodes
    "gif_path": "evaluation.gif",
    # Network parameters must match the saved model
    "conv_filters": 32,
    "conv_kernel_size": 3,
    "dense_units": 128,
}


def evaluate(config):
    """
    Evaluates the trained PPO agent.

    Args:
        config (dict): Dictionary containing evaluation configuration.
    """
    print("Initializing evaluation...")
    print(f"Configuration: {config}")

    # ##: Set random seeds for reproducibility during evaluation.
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    # ##: Initialize Environment.
    env = Maze()
    env = ImgObsWrapper(RGBImgObsWrapper(env))

    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"Observation Shape: {observation_shape}, Number of Actions: {num_actions}")

    # ##: Initialize Agent (with dummy hyperparams, only network structure matters for loading).
    network_params = {k: v for k, v in config.items() if k in ["conv_filters", "conv_kernel_size", "dense_units"]}
    # ##: Provide observation and action spaces, network params are crucial.
    agent = PPOAgent(env.observation_space, env.action_space, network_params=network_params)

    # #: Load Weights.
    load_path_prefix = Path(config["load_dir"]) / config["weights_prefix"]
    print(f"Loading weights from: {load_path_prefix}")
    agent.load_weights(load_path_prefix)

    # --- Evaluation Loop ---
    print(f"Starting evaluation for {config['num_episodes']} episodes...")
    total_rewards = []
    episode_lengths = []
    successes = 0
    frames = []

    for episode in tqdm(range(config["num_episodes"]), desc="Evaluating"):
        observation, _ = env.reset(seed=config["seed"] + episode)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if config["save_gif"]:
                frames.append(observation)

            _, action_tensor = agent.sample_action(observation)
            action = action_tensor[0].numpy()

            # ##: Environment step.
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if terminated:
                successes += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if config["save_gif"]:
            print(len(frames))
            frames.append(np.zeros_like(frames[-1]))

    # --- Results ---
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = successes / config["num_episodes"]

    print("\n--- Evaluation Results ---")
    print(f"Episodes: {config['num_episodes']}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

    if config["save_gif"] and frames:
        print(f"Saving evaluation GIF to {config['gif_path']}...")
        imageio.mimsave(config["gif_path"], frames, fps=10)
        print("GIF saved.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on MiniGrid Maze")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_EVAL_CONFIG["num_episodes"])
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_CONFIG["seed"])
    parser.add_argument("--load-dir", type=str, default=DEFAULT_EVAL_CONFIG["load_dir"])
    parser.add_argument("--weights-prefix", type=str, default=DEFAULT_EVAL_CONFIG["weights_prefix"])
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_EVAL_CONFIG["max_episode_steps"])
    parser.add_argument("--save-gif", action="store_true", default=DEFAULT_EVAL_CONFIG["save_gif"])
    parser.add_argument("--gif-path", type=str, default=DEFAULT_EVAL_CONFIG["gif_path"])
    # Network params needed if they differ from defaults used during training save
    parser.add_argument("--conv-filters", type=int, default=DEFAULT_EVAL_CONFIG["conv_filters"])
    parser.add_argument("--dense-units", type=int, default=DEFAULT_EVAL_CONFIG["dense_units"])

    args = parser.parse_args()
    config = vars(args)

    evaluate(config)
