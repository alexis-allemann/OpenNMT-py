import numpy as np

from onmt.schedulers import register_scheduler

from .scheduler import Scheduler

@register_scheduler(scheduler="tscl_online")
class TSCLOnline(Scheduler):
    """TSCL Online scheduling class."""

    def __init__(self, nb_tasks, opts) -> None:
        super().__init__(nb_tasks, opts)
        self.Q = np.zeros(nb_tasks)
        self.last_observation_by_task = np.zeros(nb_tasks)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Curriculum."""
        super().add_options(parser)
        group = parser.add_argument_group("Curriculum Learning")
        group.add_argument(
            "-tscl_smoothing",
            "--tscl_smoothing",
            type=float,
            default=0.1,
            help="Smoothing factor for the exponential moving average.",
        )
        group.add_argument(
            "-e_greedy_epsilon",
            "--e_greedy_epsilon",
            type=float,
            default=0.1,
            help="Epsilon for the epsilon-greedy policy.",
        )
        group.add_argument(
            "-tscl_policy",
            "--tscl_policy",
            type=str,
            default="epsilon_greedy",
            choices=["epsilon_greedy", "boltzmann_exploration"],
            help="Policy for the TSCL algorithm."
        )
        group.add_argument(
            "-boltzmann_temperature",
            "--boltzmann_temperature",
            type=float,
            default=1.0,
            help="Temperature for the boltzmann exploration."
        )
    
    @classmethod
    def _validate_options(cls, opts):
        super()._validate_options(opts)
        """Extra checks to validate options added from `add_options`."""
        assert 0 <= opts.tscl_smoothing <= 1, "Smoothing factor must be in [0, 1]"
        assert 0 <= opts.e_greedy_epsilon <= 1, "E-greedy epsilon must be in [0, 1]"
        assert opts.tscl_policy in ["epsilon_greedy", "boltzmann_exploration"], "TSCL policy must be in ['epsilon_greedy', 'boltzmann_exploration']"
        assert opts.boltzmann_temperature > 0, "Boltzmann temperature must be positive"

    def _parse_opts(self):
        super()._parse_opts()
        self.smoothing = self.opts.tscl_smoothing
        self.epsilon = self.opts.e_greedy_epsilon
        self.policy = self.opts.tscl_policy
        self.temperature = self.opts.boltzmann_temperature

    def init_task(self, task, reward):
        """Initialize the task."""
        self.last_observation_by_task[task] = reward
        self.Q[task] = reward * self.smoothing

    def get_starting_task(self) -> int:
        """Return the starting task."""
        return np.argmax(np.abs(self.last_observation_by_task))

    def next_task(self, step, new_reward):
        super().next_task(step, new_reward)

        reward = self.last_observation_by_task[self.current_task] - new_reward
        self.Q[self.current_task] = self.smoothing * reward + (1 - self.smoothing) * self.Q[self.current_task]

        if self.policy == "epsilon_greedy":
            return self._epsilon_greedy()
        else:
            return self._boltzmann_exploration()
    
    def _epsilon_greedy(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nb_tasks)
        else:
            return np.argmax(np.abs(self.Q))
    
    def _boltzmann_exploration(self):
        abs_q = np.abs(self.Q)
        return np.random.choice(self.nb_tasks, p=np.exp(abs_q / self.temperature) / np.sum(np.exp(abs_q / self.temperature)))