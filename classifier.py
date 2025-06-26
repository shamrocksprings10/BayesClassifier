import numpy as np
from collections import Counter

def word_frequencies(self, features: list[str]) -> dict[str, int]:
    return dict(Counter(features))

def ensure_frequencies(function):
    """
    Ensures that first argument of function (that isn't self) will
    always be a dict[str, int] even if a list[str] is passed. This
    decorator is only meant for classes inside of classifier.py.
    """
    def wrapped(self, list_arg, *args, **kwargs):
        if type(list_arg) is list:
            list_arg = word_frequencies(list_arg)
        function(self, list_arg, *args, **kwargs)
    return wrapped

class _BayesClassifier:
    def __init__(self, class_labels: list[str], num_classes: int = 2):
        self.class_labels = class_labels
        self.classes = num_classes
        self.word_probs = dict() # key = word|class, value = prob
        self.class_probs: np.ndarray[float] = np.zeros(num_classes)

    def prob_word_given_class(self, word: str, class_: str) -> float:
        return self.word_probs[f"{word}|{class_}"]

    def prob_class(self, class_: str) -> float:
        return self.class_probs[self.class_labels.index(class_)]

    @ensure_frequencies
    def prob_class_given_features(self, features: dict[str, int] | list[str], class_: str) -> float:
        # bayes theorem, remove p(x) from denominator since we only care about comparison
        prob_features_given_class = self.prob_features_given_class(features)
        return self.prob_class(class_) * prob_features_given_class

    @ensure_frequencies
    def prob_features_given_class(self, features: dict[str, int] | list[str]):
        # remove n! from multinomial PMF since we're just comparing
        return

class BayesClassifier(_BayesClassifier):
    def __init__(self, class_labels: list[str], num_classes: int = 2):
        super().__init__(self, class_labels, num_classes)