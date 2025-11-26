# apo/evaluators/bbh.py

from sklearn.metrics import accuracy_score

from apo.evaluators.registry import register_evaluator
from apo.evaluators.base import TaskEvaluator


@register_evaluator("bbh")
class BBHEvaluator(TaskEvaluator):
    """
    Evaluator for BBH Navigate Task.
    Expected logical answers: YES or NO (case-insensitive).
    """

    def parse_pred(self, output: str):
        lo = output.lower()

        if "yes" in lo:
            return "YES"
        if "no" in lo:
            return "NO"

        return None

    def normalize_label(self, label: str):
        return label.strip().upper()

    def compute_metric(self, preds, labels):
        return accuracy_score(labels, preds)