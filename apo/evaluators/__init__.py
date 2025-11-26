# apo/evaluators/__init__.py

from apo.evaluators.registry import get_evaluator, TASK_EVALUATOR_REGISTRY

# Import all evaluators to trigger registration
from apo.evaluators import binary as _  # noqa: F401
from apo.evaluators import gsm8k as _  # noqa: F401, F811
from apo.evaluators import bbh as _  # noqa: F401, F811
from apo.evaluators import wsc as _  # noqa: F401, F811

__all__ = ["get_evaluator", "TASK_EVALUATOR_REGISTRY"]
