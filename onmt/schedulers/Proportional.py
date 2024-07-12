import numpy as np
from onmt.utils.logging import logger
from onmt.schedulers import register_scheduler

from .scheduler import Scheduler


@register_scheduler(scheduler="proportional")
class Proportional(Scheduler):
    """Proportional scheduling class."""

    def __init__(self, corpora_infos, nb_actions, nb_states, opts, device_id) -> None:
        super().__init__(corpora_infos, nb_actions, nb_states, opts, device_id)
        nb_lines_array = []
        total_lines = 0
        for _, corpus in corpora_infos:
            with open(corpus["path_src"], "r") as fp:
                nb_lines = len(fp.readlines())
                nb_lines_array.append(nb_lines)
                total_lines += nb_lines
        self.tasks_prob = [nb_lines / total_lines for nb_lines in nb_lines_array]
        self.actions = range(nb_actions)

    def next_action(self, step, reward, state):
        super().next_action(step, reward, state)
        if step > self.warmup_steps:
            self.action = np.random.choice(self.actions, p=self.tasks_prob)
        else:
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
