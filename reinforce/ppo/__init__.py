"""
Proximal Policy Optimization (PPO) algorithm implementation.

This package contains the complete PPO implementation:
- agent: PPOAgent orchestrating training
- network: Actor-critic IMPALA CNN architecture
- buffer: GAE experience buffer for on-policy learning
- ppo: Core PPO loss functions and training step
- rnd: Random Network Distillation for intrinsic motivation
- exploration: Hybrid exploration strategies (epsilon-greedy, UCB, adaptive entropy)
"""
