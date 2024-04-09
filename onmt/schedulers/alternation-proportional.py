from onmt.schedulers import register_scheduler
from onmt.utils.logging import logger
from .scheduler import Scheduler
import numpy as np

@register_scheduler(scheduler="alternation-proporitonal")
class AlternationProportional(Scheduler):
    """Tasks alternation scheduling class."""
    def __init__(self, nb_tasks, opts, device_id) -> None:
        super().__init__(nb_tasks, opts, device_id)
        self.tasks_prob = [5946/627748, 182470/627748, 4509/627748, 208458/627748, 61470/627748, 103093/627748, 10017/627748, 51785/627748]
        self.actions = range(8)

    def next_task(self, step, reward):
        super().next_task(step, reward)
        self.current_task = np.random.choice(self.actions, p=self.tasks_prob)
        self._log(step)
        return self.current_task

    def _log(self, step):
        logger.info(f"Step:{step+1};GPU:{self.device_id};Task:{self.current_task}")