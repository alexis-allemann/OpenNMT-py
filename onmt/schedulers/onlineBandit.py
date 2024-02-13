import numpy as np

from onmt.schedulers import register_scheduler

from .scheduler import Scheduler

@register_scheduler(scheduler="online_bandit")
class OnlineBandit(Scheduler):
    """Online Bandit scheduling class."""

    def __init__(self, nb_tasks, starting_task, opts) -> None:
        super().__init__(nb_tasks, starting_task, opts)
        self.Q = np.zeros(nb_tasks)
        self.current_xent = 0
        self.visited_tasks = []
        self.nb_init_iterations = 0

    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Curriculum."""
        super().add_options(parser)
        group = parser.add_argument_group("Curriculum Learning")
        group.add_argument(
            "-bandit_learning_rate",
            "--bandit_learning_rate",
            type=float,
            default=0.1,
            help="Learning rate for the bandit algorithm.",
        )
        group.add_argument(
            "-bandit_epsilon",
            "--bandit_epsilon",
            type=float,
            default=0.1,
            help="Epsilon for the bandit algorithm.",
        )
        group.add_argument(
            "-bandit_policy",
            "--bandit_policy",
            type=str,
            default="epsilon_greedy",
            choices=["epsilon_greedy", "boltzmann_exploration"],
            help="Policy for the bandit algorithm."
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
        assert 0 <= opts.bandit_learning_rate <= 1, "Bandit learning rate must be in [0, 1]"
        assert 0 <= opts.bandit_epsilon <= 1, "Bandit epsilon must be in [0, 1]"
        assert opts.bandit_policy in ["epsilon_greedy", "boltzmann_exploration"], "Bandit policy must be in ['epsilon_greedy', 'boltzmann_exploration']"
        assert opts.boltzmann_temperature > 0, "Boltzmann temperature must be positive"

    def _parse_opts(self):
        super()._parse_opts()
        self.learning_rate = self.opts.bandit_learning_rate
        self.epsilon = self.opts.bandit_epsilon
        self.policy = self.opts.bandit_policy
        self.temperature = self.opts.boltzmann_temperature

    def next_task(self, step, model, optim, stats):
        super().next_task(step, model, optim, stats)

        if len(self.visited_tasks) < self.nb_tasks:
            if self.nb_init_iterations < 1:
                self.nb_init_iterations += 1
                self.current_xent = stats.xent()
                return self.current_task
            else:
                self.visited_tasks.append(self.current_task)
                self.Q[self.current_task] = self.current_xent - stats.xent()
                self.current_task = (self.current_task + 1) % self.nb_tasks
                self.nb_init_iterations = 0

        xent_diff = self.current_xent - stats.xent()
        self.update(self.current_task, xent_diff)
        if self.policy == "epsilon_greedy":
            return self._epsilon_greedy()
        else:
            return self._boltzmann_exploration()
    
    def _epsilon_greedy(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nb_tasks)
        else:
            return np.argmax(self.Q)
    
    def _boltzmann_exploration(self):
        return np.random.choice(self.nb_tasks, p=np.exp(self.Q / self.temperature) / np.sum(np.exp(self.Q / self.temperature)))
    
    def update(self, task_index, reward):
        self.Q[task_index] = self.learning_rate * reward + (1 - self.learning_rate) * self.Q[task_index]