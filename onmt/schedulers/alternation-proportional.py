from onmt.schedulers import register_scheduler
from onmt.utils.logging import logger
from .scheduler import Scheduler
import numpy as np

@register_scheduler(scheduler="alternation-proporitonal")
class AlternationProportional(Scheduler):
    """Tasks alternation scheduling class."""
    def __init__(self, nb_actions, nb_states, opts, device_id) -> None:
        super().__init__(nb_actions, nb_states, opts, device_id)
        self.tasks_prob = [5946/627748, 182470/627748, 4509/627748, 208458/627748, 61470/627748, 103093/627748, 10017/627748, 51785/627748]
        self.actions = range(8)

    def next_task(self, step, reward, state):
        super().next_task(step, reward, state)
        if step > self.warmup_steps:
            self.current_task = np.random.choice(self.actions, p=self.tasks_prob)
        else:
            hrl_actions = [1,3,5,7]
            self.current_task = np.random.choice(hrl_actions)
        self._log(step)
        return self.current_task

    def _log(self, step):
        logger.info(f"Step:{step+1};GPU:{self.device_id};Task:{self.current_task}")

    def needs_reward(self):
        return False
    
    def needs_state(self):
        return False