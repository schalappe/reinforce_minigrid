"""
Learning rate and exploration schedules for RL training.

Provides schedule classes for:
- Learning rate annealing (PPO)
- Epsilon decay (DQN exploration)
"""


class LinearSchedule:
    """
    Linear interpolation between start and end values.

    Used for learning rate decay and epsilon-greedy exploration.
    """

    def __init__(self, start: float, end: float, total_steps: int):
        """
        Initialize linear schedule.

        Parameters
        ----------
        start : float
            Starting value.
        end : float
            Final value (reached at total_steps).
        total_steps : int
            Number of steps over which to interpolate.
        """
        self.start = start
        self.end = end
        self.total_steps = max(1, total_steps)

    def value(self, step: int) -> float:
        """
        Get scheduled value at given step.

        Parameters
        ----------
        step : int
            Current step.

        Returns
        -------
        float
            Interpolated value between start and end.
        """
        progress = min(1.0, step / self.total_steps)
        return self.start + progress * (self.end - self.start)

    def __call__(self, step: int) -> float:
        """Alias for value() method."""
        return self.value(step)
