from onmt.schedulers import register_scheduler

from .scheduler import Scheduler


@register_scheduler(scheduler="alternation")
class Alternation(Scheduler):
    """Tasks alternation scheduling class."""

    def next_task(self, step, reward, state):
        super().next_task(step, reward, state)
        self.current_task = (self.current_task + 1) % self.nb_actions
        return self.current_task
