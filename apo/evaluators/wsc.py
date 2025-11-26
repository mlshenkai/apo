# apo/evaluators/wsc.py

from sklearn.metrics import accuracy_score

from apo.evaluators.registry import register_evaluator
from apo.evaluators.base import TaskEvaluator


@register_evaluator("wsc")
class WSCEvaluator(TaskEvaluator):
    """
    Evaluator for Winograd Schema Challenge (WSC).
    Expect answers: "A" or "B".
    """

    def parse_pred(self, output: str):
        lo = output.lower().strip()

        # direct forms
        if lo == "a" or "option a" in lo:
            return "A"
        if lo == "b" or "option b" in lo:
            return "B"

        # fallback: search last A/B
        for c in reversed(output.strip()):
            if c.upper() in ["A", "B"]:
                return c.upper()

        return None

    def normalize_label(self, label: str):
        """
        Clean and normalize gold label to "A" or "B".
        """
        return label.strip().upper()

    def compute_metric(self, preds, labels):
        return accuracy_score(labels, preds)