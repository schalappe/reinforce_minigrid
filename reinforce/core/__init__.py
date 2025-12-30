"""
Core shared utilities for reinforcement learning algorithms.

This module provides base classes and shared utilities used by both PPO and DQN implementations:
- BaseAgent: Abstract base class for all RL agents
- BaseBuffer: Abstract base class for experience buffers
- network_utils: Shared network building blocks (orthogonal init, CNN layers)
- schedules: Learning rate and exploration schedules
- normalization: Running statistics for observation/reward normalization
"""
