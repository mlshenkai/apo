from apo.evaluators.base import TaskEvaluator

TASK_EVALUATOR_REGISTRY = {}

def register_evaluator(task_name: str):
    def wrapper(cls):
        TASK_EVALUATOR_REGISTRY[task_name] = cls()
        return cls
    return wrapper


def get_evaluator(task: str) -> TaskEvaluator:
    if task not in TASK_EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator not found for task '{task}'")
    return TASK_EVALUATOR_REGISTRY[task]