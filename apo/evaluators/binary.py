# apo/evaluators/binary.py

from sklearn.metrics import f1_score

from apo.evaluators.registry import register_evaluator
from apo.evaluators.base import TaskEvaluator


@register_evaluator("liar")
@register_evaluator("ethos")
@register_evaluator("arsarcasm")
class BinaryEvaluator(TaskEvaluator):
    """
    Generic evaluator for binary Yes/No tasks.
    """

    def parse_pred(self, output: str):
        lo = output.lower().strip()

        if "yes" in lo:
            return "Yes"
        if "no" in lo:
            return "No"

        # fallback: exact match
        if lo in ["yes", "no"]:
            return lo.capitalize()

        return output.strip()

    def normalize_label(self, label: str):
        """
        Ensure gold label is exactly "Yes" or "No".
        """
        return label.strip().capitalize()

    def compute_metric(self, preds, labels):
        return f1_score(labels, preds, average="macro")