"""Tests for PrioritizedReplayBuffer n-step handling."""

import numpy as np
import pytest

from reinforce.dqn.buffer import PrioritizedReplayBuffer


class TestNStepTerminalHandling:
    """Tests for correct n-step terminal state handling."""

    @pytest.fixture
    def buffer(self) -> PrioritizedReplayBuffer:
        """Create buffer with n_step=3 for testing."""
        return PrioritizedReplayBuffer(
            obs_shape=(4, 4, 3),
            capacity=100,
            n_step=3,
            gamma=0.99,
        )

    def _make_state(self, value: float) -> np.ndarray:
        """Create a state filled with a specific value for easy identification."""
        return np.full((4, 4, 3), value, dtype=np.float32)

    def test_nstep_uses_terminal_next_state_when_early_done(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        When a terminal occurs before n_step, next_state should come from
        the terminal transition, not the last buffer element.

        Scenario: n_step=3, buffer receives [t0, t1(done=True), t2]
        Expected: stored next_state should be t1's next_state, not t2's.
        """
        # ##>: Create distinguishable states for each transition.
        state_0 = self._make_state(0.0)
        state_1 = self._make_state(1.0)
        state_2 = self._make_state(2.0)
        next_state_0 = self._make_state(10.0)
        next_state_1 = self._make_state(11.0)  # Terminal next_state.
        next_state_2 = self._make_state(12.0)  # Should NOT be used.

        # ##>: Store 3 transitions; second one is terminal.
        buffer.store(state_0, action=0, reward=1.0, next_state=next_state_0, done=False)
        buffer.store(state_1, action=1, reward=2.0, next_state=next_state_1, done=True)
        buffer.store(state_2, action=2, reward=3.0, next_state=next_state_2, done=False)

        # ##>: After 3 stores, buffer should have 1 transition.
        assert len(buffer) == 1

        # ##>: Verify stored next_state is from terminal (t1), not last (t2).
        stored_next_state = buffer.next_states[0]
        assert np.allclose(stored_next_state, next_state_1), (
            'next_state should be from terminal transition (value=11), '
            f'got mean={stored_next_state.mean():.1f}'
        )

        # ##>: Verify done flag is True (from terminal).
        assert buffer.dones[0]

    def test_nstep_uses_terminal_done_flag(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        When terminal occurs early, done flag should be True even if
        subsequent transitions have done=False.
        """
        state = self._make_state(0.0)
        next_state = self._make_state(1.0)

        buffer.store(state, action=0, reward=1.0, next_state=next_state, done=False)
        buffer.store(state, action=0, reward=1.0, next_state=next_state, done=True)
        buffer.store(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert buffer.dones[0]

    def test_nstep_return_truncated_at_terminal(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        N-step return should only sum rewards up to and including terminal.

        Scenario: n_step=3, rewards=[1.0, 2.0, 3.0], terminal at step 1
        Expected return: 1.0 + 0.99 * 2.0 = 2.98 (not including reward 3.0)
        """
        state = self._make_state(0.0)
        next_state = self._make_state(1.0)
        gamma = 0.99

        buffer.store(state, action=0, reward=1.0, next_state=next_state, done=False)
        buffer.store(state, action=0, reward=2.0, next_state=next_state, done=True)
        buffer.store(state, action=0, reward=3.0, next_state=next_state, done=False)

        expected_return = 1.0 + gamma * 2.0  # 2.98
        stored_reward = buffer.rewards[0]

        assert np.isclose(stored_reward, expected_return, atol=1e-5), (
            f'Expected n-step return {expected_return:.4f}, got {stored_reward:.4f}'
        )

    def test_nstep_no_terminal_uses_last_element(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        When no terminal occurs, next_state should come from last element.
        """
        next_state_0 = self._make_state(10.0)
        next_state_1 = self._make_state(11.0)
        next_state_2 = self._make_state(12.0)  # Should be used.
        state = self._make_state(0.0)

        buffer.store(state, action=0, reward=1.0, next_state=next_state_0, done=False)
        buffer.store(state, action=0, reward=1.0, next_state=next_state_1, done=False)
        buffer.store(state, action=0, reward=1.0, next_state=next_state_2, done=False)

        stored_next_state = buffer.next_states[0]
        assert np.allclose(stored_next_state, next_state_2), (
            'Without terminal, next_state should be from last element (value=12), '
            f'got mean={stored_next_state.mean():.1f}'
        )
        assert not buffer.dones[0]


class TestFlushNStepBuffer:
    """Tests for flush_n_step_buffer terminal handling."""

    @pytest.fixture
    def buffer(self) -> PrioritizedReplayBuffer:
        """Create buffer with n_step=3 for testing."""
        return PrioritizedReplayBuffer(
            obs_shape=(4, 4, 3),
            capacity=100,
            n_step=3,
            gamma=0.99,
        )

    def _make_state(self, value: float) -> np.ndarray:
        """Create a state filled with a specific value for easy identification."""
        return np.full((4, 4, 3), value, dtype=np.float32)

    def test_flush_uses_terminal_next_state(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        Flush should use terminal's next_state when terminal occurs early.
        """
        state = self._make_state(0.0)
        next_state_terminal = self._make_state(11.0)

        # ##>: Store only 2 transitions (less than n_step=3).
        buffer.store(state, action=0, reward=1.0, next_state=self._make_state(10.0), done=False)
        buffer.store(state, action=1, reward=2.0, next_state=next_state_terminal, done=True)

        assert len(buffer) == 0  # Not enough for n_step yet.

        # ##>: Flush remaining transitions.
        buffer.flush_n_step_buffer()

        assert len(buffer) == 2  # Both transitions should be stored.

        # ##>: First stored transition should use terminal's next_state.
        stored_next_state = buffer.next_states[0]
        assert np.allclose(stored_next_state, next_state_terminal), (
            'Flushed next_state should be from terminal (value=11), '
            f'got mean={stored_next_state.mean():.1f}'
        )

    def test_flush_truncates_return_at_terminal(
        self, buffer: PrioritizedReplayBuffer
    ) -> None:
        """
        Flush should truncate n-step return at terminal.
        """
        state = self._make_state(0.0)
        next_state = self._make_state(1.0)
        gamma = 0.99

        buffer.store(state, action=0, reward=1.0, next_state=next_state, done=False)
        buffer.store(state, action=0, reward=2.0, next_state=next_state, done=True)

        buffer.flush_n_step_buffer()

        # ##>: First transition: 1.0 + 0.99 * 2.0 = 2.98.
        expected_return = 1.0 + gamma * 2.0
        assert np.isclose(buffer.rewards[0], expected_return, atol=1e-5)

        # ##>: Second transition: just 2.0 (terminal itself).
        assert np.isclose(buffer.rewards[1], 2.0, atol=1e-5)
