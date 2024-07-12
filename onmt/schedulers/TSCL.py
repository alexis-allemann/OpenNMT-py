import numpy as np

from onmt.schedulers import register_scheduler
from onmt.utils.logging import logger
from .scheduler import Scheduler


@register_scheduler(scheduler="tscl")
class TSCL(Scheduler):
    """TSCL scheduling class."""

    def __init__(self, copora_infos, nb_actions, nb_states, opts, device_id) -> None:
        super().__init__(copora_infos, nb_actions, nb_states, opts, device_id)
        self.Q = np.zeros(nb_actions)
        self.last_observation_by_action = np.zeros(nb_actions)
        self.unvisited_actions = list(range(nb_actions))
        self.action = 0

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
            "-tscl_eps",
            "--tscl_eps",
            type=float,
            default=0.1,
            help="Epsilon for the epsilon-greedy policy.",
        )

    @classmethod
    def _validate_options(cls, opts):
        super()._validate_options(opts)
        """Extra checks to validate options added from `add_options`."""
        assert 0 <= opts.tscl_smoothing <= 1, "Smoothing factor must be in [0, 1]"
        assert 0 <= opts.tscl_eps <= 1, "E-greedy epsilon must be in [0, 1]"

    def _parse_opts(self):
        super()._parse_opts()
        self.smoothing = self.opts.tscl_smoothing
        self.epsilon = self.opts.tscl_eps

    def needs_state(self):
        return False

    def next_action(self, step, new_reward, state):
        super().next_action(step, new_reward, state)

        reward = self.last_observation_by_action[self.action] - new_reward
        self.last_observation_by_action[self.action] = new_reward
        self.Q[self.action] = (
            self.smoothing * reward + (1 - self.smoothing) * self.Q[self.action]
        )

        if len(self.unvisited_actions) > 0:
            logger.info(f"Unvisited actions: {self.unvisited_actions}")
            action = np.random.choice(self.unvisited_actions)
            self.unvisited_actions.remove(action)
        else:
            if step < self.warmup_steps:
                logger.info("Warmup step - Random action selection.")
                available_actions = list(range(self.nb_actions))
                if self.hrl_warmup:
                    available_actions = self.hrl_warmup_actions
                action = np.random.choice(available_actions)
            else:
                action = self._epsilon_greedy()

        self.action = action
        self._log(step)
        return action

    def _epsilon_greedy(self):
        if np.random.rand() < self.epsilon:
            logger.info("E-greedy choice - Random action selection.")
            return np.random.choice(self.nb_actions)
        else:
            return np.argmax(np.abs(self.Q))

    def _log(self, step):
        qvalues = "["
        for i in range(self.nb_actions):
            qvalues += f"{self.Q[i]}"
            if i < self.nb_actions - 1:
                qvalues += ", "
        qvalues += "]"
        logger.info(
            f"Step:{step+1};GPU:{self.device_id};Q-values:{qvalues};Action:{self.action+1}"
        )
