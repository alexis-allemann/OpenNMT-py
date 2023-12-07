from onmt.schedulers import register_scheduler

from .scheduler import Scheduler


@register_scheduler(scheduler="alternation")
class Alternation(Scheduler):
    """Tasks alternation scheduling class."""

    def next_task(self, step, model, optim):
        super().next_task(step, model, optim)
        self.current_task = (self.current_task + 1) % self.nb_tasks
        return self.current_task
