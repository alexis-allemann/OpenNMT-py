from onmt.utils.logging import logger
from onmt.schedulers import register_scheduler

from .scheduler import Scheduler


@register_scheduler(scheduler="alternation")
class Alternation(Scheduler):
    """Tasks alternation scheduling class."""

    def next_task(self, step, reward):
        super().next_task(step, reward)
        self.current_task = (self.current_task + 1) % self.nb_tasks
        self._log(step)
        return self.current_task

    def _log(self, step):
        logger.info(f"Step:{step+1};GPU:{self.device_id};Task:{self.current_task}")