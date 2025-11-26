from abc import ABC, abstractmethod

class TaskEvaluator(ABC):

    @abstractmethod
    def parse_pred(self, output: str):
        pass

    def normalize_label(self, label: str):
        return label

    @abstractmethod
    def compute_metric(self, preds, labels):
        pass