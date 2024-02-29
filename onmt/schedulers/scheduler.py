"""Base scheduler class and relate utils."""

from onmt.utils.logging import logger


class Scheduler(object):
    """A Base class that every curriculum scheduler method should derived from."""

    def __init__(self, nb_tasks, opts, device_id) -> None:
        self.nb_tasks = nb_tasks
        self.current_task = 0
        self.opts = opts
        self.device_id = device_id
        self._parse_opts()
    
    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Transform."""
        pass

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks to validate options added from `add_options`."""
        pass
    
    def _parse_opts(self):
        """Parse opts to set/reset instance's attributes.

        This should be override if there are attributes other than self.opts.
        To make sure we recover from picked state.
        """
        pass

    def init_task(self, task, reward):
        """Initialize the task."""
        pass

    def get_starting_task(self) -> int:
        """Return the starting task."""
        pass
    
    def next_task(self, step, reward):
        logger.info(f"Step: {step} - Next task scheduling")
