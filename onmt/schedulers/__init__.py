"""Module for curriculum learning schedulers"""
import os
import importlib

from .scheduler import Scheduler

AVAILABLE_SCHEDULERS = {}


def get_scheduler_cls(scheduler_name):
    """Returns scheduler related to the name
    indicated in `scheduler_name`."""
    if scheduler_name not in AVAILABLE_SCHEDULERS.keys():
        raise ValueError("specified scheduler not supported!")
    return AVAILABLE_SCHEDULERS[scheduler_name]


__all__ = ["get_scheduler_cls"]


def register_scheduler(scheduler):
    """Scheduler register that can be used to add new scheduling class."""

    def register_schedulers_cls(cls):
        if scheduler in AVAILABLE_SCHEDULERS.keys():
            raise ValueError(
                "Cannot register duplicate scheduler named ({})".format(scheduler)
            )
        if not issubclass(cls, Scheduler):
            raise ValueError(
                "Scheduler ({}: {}) must extend Scheduler".format(scheduler, cls.__name__)
            )
        AVAILABLE_SCHEDULERS[scheduler] = cls
        return cls

    return register_schedulers_cls


# Auto import python files in this directory
schedulers_dir = os.path.dirname(__file__)
for file in os.listdir(schedulers_dir):
    path = os.path.join(schedulers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        file_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("onmt.schedulers." + file_name)
