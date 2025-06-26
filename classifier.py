import numpy as np
from scipy.special import factorial
from collections import Counter

def word_frequencies(features: list[str]) -> dict[str, int]:
    return dict(Counter(features))

def ensure_frequencies(function):
    """
    Ensures that first argument of function (that isn't self) will
    always be a dict[str, int] even if a list[str] is passed. This
    decorator is only meant for classes inside of classifier.py.
    """
    def wrapped(self, list_arg: dict[str, int] | list[str], *args, **kwargs):
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
    def prob_class_given_features(self, features: dict[str, int], class_: str) -> float:
        # bayes theorem, remove p(x) from denominator since we only care about comparison
        return self.prob_class(class_) * self.prob_features_given_class(features, class_)

    @ensure_frequencies
    def prob_features_given_class(self, features: dict[str, int], class_: str) -> float:
        # remove n! from multinomial PMF since we're just comparing
        words = features.keys()
        counts = np.array(features.values())

        factor = 1 / np.prod(factorial(counts))
        prob = np.prod(np.array([self.prob_word_given_class(words[i], class_) ** c_i for i, c_i in enumerate(counts)]))
        return factor * prob

    @ensure_frequencies
    def prob_classes_given_features(self, features: dict[str, int]):
        return np.array([self.prob_class_given_features(features, class_) for class_ in self.num_classes])

    def train(self):
        pass

class BayesClassifier(_BayesClassifier):
    def __init__(self, class_labels: list[str], num_classes: int = 2):
        super().__init__(class_labels, num_classes)