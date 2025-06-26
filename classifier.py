import numpy as np
import pandas as pd
from collections import Counter

factorial_cache = {1 : 1}
def factorial(x: int) -> int:
    if x in factorial_cache.keys():
        return factorial_cache[x]
    answer = x * factorial(x - 1)
    factorial_cache[x] = answer
    return answer

def flatten(li):
    return_val = []
    for l in li:
        return_val += l
    return return_val

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
        return function(self, list_arg, *args, **kwargs)
    return wrapped

class _BayesClassifier:
    def __init__(self, class_labels: list[str], num_classes: int = 2):
        self.class_labels = class_labels
        self.num_classes = num_classes
        self.class_probs: dict[str, float] = {class_: 0 for class_ in class_labels}
        self.word_probs = dict()  # key = word|class, value = prob

    def prob_word_given_class(self, word: str, class_: str) -> float:
        return self.word_probs[f"{word}|{class_}"]

    def prob_class(self, class_: str) -> float:
        return self.class_probs[class_]

    def prob_class_given_features(self, features: dict[str, int], class_: str) -> float:
        # bayes theorem, remove p(x) from denominator since we the only care about comparison
        return self.prob_class(class_) * self.prob_features_given_class(features, class_)

    @ensure_frequencies
    def prob_features_given_class(self, features: dict[str, int], class_: str) -> float:
        # remove n! from multinomial PMF since we're just comparing
        features = {key : value for key, value in features.items() if f"{key}|{class_}" in self.word_probs.keys()}

        words = list(features.keys())
        counts = list(features.values())

        factor = 1 / np.prod([factorial(count) for count in counts])
        prob = np.prod(np.array([self.prob_word_given_class(words[i], class_) ** c_i for i, c_i in enumerate(counts)]))
        return factor * prob

    def prob_classes_given_features(self, features: dict[str, int]):
        return np.array([self.prob_class_given_features(features, class_) for class_ in self.class_labels])

    def train(self, train_df: pd.DataFrame):
        # prepare class_probs
        class_probs = train_df["class"].value_counts(normalize=True)
        for i, class_label in enumerate(self.class_labels):
            self.class_probs[i] = class_probs[class_label]

        counters = [Counter() for _ in range(self.num_classes)]
        for i, row in train_df.iterrows():
            class_ = row["class"]
            class_index = self.class_labels.index(class_)
            counters[class_index].update(row["words"])

        for i, class_ in enumerate(self.class_labels):
            counter = counters[i]
            for word in counter.keys():
                self.word_probs[f"{word}|{class_}"] = counter[word] / counter.total()

    def evaluate(self, *features):
        return [self.prob_classes_given_features(feature_vector) for feature_vector in features]

    def test(self, test_df):
        evaluations = self.evaluate(*test_df["words"].to_list())
        predicted_class = list(map(lambda probs: self.class_labels[int(np.argmax(probs))], evaluations))
        eval_classes = pd.Series(predicted_class)
        test_classes = test_df["class"].reset_index(drop=True)
        return (eval_classes == test_classes).value_counts()

class BayesClassifier(_BayesClassifier):
    def __init__(self, class_labels: list[str], num_classes: int = 2):
        super().__init__(class_labels, num_classes)