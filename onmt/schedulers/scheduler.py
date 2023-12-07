"""Base scheduler class and relate utils."""

from onmt.utils.logging import logger


class Scheduler(object):
    """A Base class that every curriculum scheduler method should derived from."""

    def __init__(self, nb_tasks, starting_task) -> None:
        self.nb_tasks = nb_tasks
        self.current_task = starting_task

    def next_task(self, step, model, optim):
        logger.info(f"Step: {step} - Next task scheduling")
