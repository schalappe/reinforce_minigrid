"""
Prioritized Experience Replay Buffer for Rainbow DQN.

Implements:
- Sum tree for efficient priority-based sampling
- Multi-step transition storage
- Importance sampling weights
"""

import numpy as np

from reinforce.core.base_buffer import BaseBuffer


class SumTree:
    """
    Binary sum tree for O(log n) priority sampling.

    Each leaf stores a priority. Parent nodes store sum of children.
    Allows efficient priority updates and proportional sampling.
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree.

        Parameters
        ----------
        capacity : int
            Maximum number of leaves (transitions).
        """
        self.capacity = capacity
        # ##>: Tree has 2*capacity - 1 nodes (leaves + internal nodes).
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def add(self, priority: float) -> int:
        """
        Add priority and return data index.

        Parameters
        ----------
        priority : float
            Priority value for the new transition.

        Returns
        -------
        int
            Data index where the transition should be stored.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority at tree index and propagate to root.

        Parameters
        ----------
        tree_idx : int
            Index in the tree array.
        priority : float
            New priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # ##>: Propagate change up to root.
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> tuple[int, float, int]:
        """
        Sample leaf proportional to priority.

        Parameters
        ----------
        value : float
            Random value in [0, total_priority).

        Returns
        -------
        tuple[int, float, int]
            (tree_idx, priority, data_idx).
        """
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    @property
    def total(self) -> float:
        """Return total priority (root node value)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Return maximum priority in the tree."""
        return float(np.max(self.tree[self.capacity - 1 :]))


class PrioritizedReplayBuffer(BaseBuffer):
    """
    Prioritized Experience Replay with multi-step returns.

    Parameters
    ----------
    obs_shape : tuple
        Observation shape.
    capacity : int
        Maximum buffer size.
    alpha : float, optional
        Priority exponent (0 = uniform, 1 = full prioritization). Default is 0.6.
    beta_start : float, optional
        Initial importance sampling exponent. Default is 0.4.
    beta_end : float, optional
        Final importance sampling exponent. Default is 1.0.
    beta_frames : int, optional
        Frames to anneal beta. Default is 100_000.
    n_step : int, optional
        Number of steps for multi-step returns. Default is 3.
    gamma : float, optional
        Discount factor. Default is 0.99.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__(obs_shape, capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.n_step = n_step
        self.gamma = gamma

        # ##>: Storage arrays.
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # ##>: Priority tree.
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.ptr = 0
        self.size = 0
        self.frame = 0

        # ##>: Multi-step buffer (temporary storage).
        self.n_step_buffer: list[tuple] = []

    @property
    def has_pending_n_step_transitions(self) -> bool:
        """Return True if n-step buffer has pending transitions to flush."""
        return len(self.n_step_buffer) > 0

    def _get_beta(self) -> float:
        """Get current beta for importance sampling."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def store(  # type: ignore[override]
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition with multi-step handling.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next observation.
        done : bool
            Whether episode ended.
        """
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # ##>: Compute n-step return and track terminal index.
        n_step_return = 0.0
        terminal_idx = len(self.n_step_buffer) - 1
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma**i) * r
            if d:
                terminal_idx = i
                break

        state_0 = self.n_step_buffer[0][0]
        action_0 = self.n_step_buffer[0][1]
        next_state_n = self.n_step_buffer[terminal_idx][3]
        done_n = self.n_step_buffer[terminal_idx][4]

        idx = self.tree.add(self.max_priority**self.alpha)
        self.states[idx] = state_0
        self.actions[idx] = action_0
        self.rewards[idx] = n_step_return
        self.next_states[idx] = next_state_n
        self.dones[idx] = done_n

        self.size = min(self.size + 1, self.capacity)
        self.n_step_buffer.pop(0)

    def flush_n_step_buffer(self) -> None:
        """Flush remaining transitions in n-step buffer (call on episode end)."""
        while len(self.n_step_buffer) > 0:
            # ##>: Compute n-step return and track terminal index.
            n_step_return = 0.0
            terminal_idx = len(self.n_step_buffer) - 1
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma**i) * r
                if d:
                    terminal_idx = i
                    break

            state_0 = self.n_step_buffer[0][0]
            action_0 = self.n_step_buffer[0][1]
            next_state_n = self.n_step_buffer[terminal_idx][3]
            done_n = self.n_step_buffer[terminal_idx][4]

            idx = self.tree.add(self.max_priority**self.alpha)
            self.states[idx] = state_0
            self.actions[idx] = action_0
            self.rewards[idx] = n_step_return
            self.next_states[idx] = next_state_n
            self.dones[idx] = done_n

            self.size = min(self.size + 1, self.capacity)
            self.n_step_buffer.pop(0)

    def store_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Store a batch of transitions efficiently (vectorized).

        Bypasses n-step processing for simplicity in batch mode.
        Use this for vectorized environment collection.

        Parameters
        ----------
        states : np.ndarray
            Batch of observations, shape (batch, *obs_shape).
        actions : np.ndarray
            Batch of actions, shape (batch,).
        rewards : np.ndarray
            Batch of rewards, shape (batch,).
        next_states : np.ndarray
            Batch of next observations, shape (batch, *obs_shape).
        dones : np.ndarray
            Batch of done flags, shape (batch,).
        """
        batch_size = len(states)

        for i in range(batch_size):
            idx = self.tree.add(self.max_priority**self.alpha)
            self.states[idx] = states[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_states[idx] = next_states[i]
            self.dones[idx] = dones[i]

        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample batch with prioritized sampling (vectorized).

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        tuple
            (states, actions, rewards, next_states, dones, indices, weights).
        """
        self.frame += batch_size
        beta = self._get_beta()

        # ##>: Vectorized stratified sampling for better coverage.
        segment = self.tree.total / batch_size
        segment_starts = np.arange(batch_size) * segment
        random_offsets = np.random.uniform(0, segment, size=batch_size)
        values = segment_starts + random_offsets

        # ##>: Batch tree traversal (still requires loop but optimized).
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)
        tree_indices = np.zeros(batch_size, dtype=np.int32)

        for i, value in enumerate(values):
            tree_idx, priority, data_idx = self.tree.get(value)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = max(priority, 1e-8)

        # ##>: Vectorized importance sampling weights.
        total_priority = self.tree.total + 1e-8
        probs = priorities / total_priority
        weights = np.power(self.size * probs, -beta)
        weights /= weights.max() + 1e-8

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            tree_indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Parameters
        ----------
        tree_indices : np.ndarray
            Tree indices of sampled transitions.
        td_errors : np.ndarray
            Absolute TD errors for priority update.
        """
        priorities = np.abs(td_errors) + 1e-6
        self.max_priority = max(self.max_priority, float(priorities.max()))

        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority) ** self.alpha)

    def clear(self) -> None:
        """Reset buffer."""
        self.ptr = 0
        self.size = 0
        self.tree = SumTree(self.capacity)
        self.n_step_buffer.clear()
        self.max_priority = 1.0
        self.frame = 0

    def __len__(self) -> int:
        return self.size
