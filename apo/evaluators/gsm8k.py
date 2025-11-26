# apo/evaluators/gsm8k.py

import re
from sklearn.metrics import accuracy_score

from apo.evaluators.registry import register_evaluator
from apo.evaluators.base import TaskEvaluator


@register_evaluator("gsm8k")
class GSM8KEvaluator(TaskEvaluator):
    """
    Evaluator for GSM8K (math word problems with integer answers).
    """
    
    def parse_pred(self, output: str):
        """
        Extract the last integer from model output.
        Example: "The answer is 24." → "24"
        """
        text = output.replace(",", "")
        nums = re.findall(r"-?\d+", text)
        return nums[-1] if nums else None

    def normalize_label(self, label: str):
        """
        Normalize gold answer into canonical integer string format.
        """
        return str(int(label))

    def compute_metric(self, preds, labels):
        return accuracy_score(labels, preds)