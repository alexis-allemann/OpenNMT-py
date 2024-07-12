import numpy as np
from onmt.utils.logging import logger
from onmt.schedulers import register_scheduler

from .scheduler import Scheduler


@register_scheduler(scheduler="uniform")
class Uniform(Scheduler):
    """Uniform scheduling class."""

    def next_action(self, step, reward, state):
        super().next_action(step, reward, state)
        available_actions = list(range(self.nb_actions))
        if self.hrl_warmup:
            available_actions = self.hrl_warmup_actions
        self.action = np.random.choice(available_actions)
        self._log(step)
        return self.action

    def _log(self, step):
        logger.info(f"Step:{step+1};GPU:{self.device_id};Action:{self.action+1}")

    def needs_reward(self):
        return False

    def needs_state(self):
        return False
