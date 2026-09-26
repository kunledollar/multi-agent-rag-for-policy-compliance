"""Dataset-governed Sentinel evaluation framework.

Evaluation models are loaded lazily so production graph utilities do not require
the evaluation dependency stack merely to use refusal detection.
"""

__all__ = ["ExecutionMode"]


def __getattr__(name):
    if name == "ExecutionMode":
        from .models import ExecutionMode
        return ExecutionMode
    raise AttributeError(name)
