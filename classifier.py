import numpy as np
import pandas as pd
from functools import reduce
from operator import mul, add
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
    """
    Simplifications made to the model, these don't compromise in classification.
    Usage of log probabilities so float underflow can't happen.
    Scaling factors such as p(X_vector) in prob_class_given_features() and n!
    in prob_features_given_class().
    Since we only care about the maximum probability for Class i, these changes won't
    affect the ordering of p(C_i | X_vector).
    """
    def __init__(self, class_labels: list[str]):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.class_probs: dict[str, float] = {class_: 0 for class_ in class_labels}
        self.word_probs = {class_ : dict() for class_ in self.class_labels}  # [class][word] = prob

    def prob_class_given_features(self, features: dict[str, int], class_: str) -> float:
        # bayes theorem, remove p(x) from denominator since we the only care about comparison
        return self.class_probs[class_] * self.prob_features_given_class(features, class_)

    @ensure_frequencies
    def prob_features_given_class(self, features: dict[str, int], class_: str) -> float:
        # remove n! from multinomial PMF since we're just comparing
        features = {word : value for word, value in features.items() if word in self.word_probs[class_].keys() and value > 0}
        if len(features.keys()) == 0: # quite unique to have no words in common with training data
            return 0

        words = list(features.keys())
        counts = list(features.values())

        factor = 1 / reduce(mul, [factorial(count) for count in counts])
        prob = np.sum([self.word_probs[class_][words[i]] * c_i for i, c_i in enumerate(counts)])
        return factor * prob

    def prob_classes_given_features(self, features: dict[str, int]):
        return np.array([
            self.prob_class_given_features(features, class_) for class_ in self.class_labels])

    def train(self, train_df: pd.DataFrame):
        # prepare class_probs
        class_probs = train_df["class"].value_counts(normalize=True)
        self.class_probs = {class_label: class_probs[class_label] for class_label in self.class_labels}

        counters = [Counter() for _ in range(self.num_classes)]
        for i, row in train_df.iterrows():
            class_ = row["class"]
            class_index = self.class_labels.index(class_)
            counters[class_index].update(row["words"])

        for i, class_ in enumerate(self.class_labels):
            counter = counters[i]
            for word in counter.keys():
                # prob = counter[word] / counter.total()
                prob = 1 / len(set(counter.keys()))
                self.word_probs[class_][word] = np.log(prob)

    def evaluate(self, *features):
        return [self.prob_classes_given_features(feature_vector) for feature_vector in features]

    def test(self, test_df, accuracy_only=False):
        evaluations = self.evaluate(*test_df["words"].to_list())
        predicted_class = list(map(lambda probs: self.class_labels[int(np.argmax(probs))], evaluations))
        eval_classes = pd.Series(predicted_class)
        if accuracy_only:
            test_classes = test_df["class"].reset_index(drop=True)
            print(eval_classes, test_classes)
            return (eval_classes == test_classes).value_counts()
        return eval_classes

class BayesClassifier(_BayesClassifier):
    def __init__(self, class_labels: list[str]):
        super().__init__(class_labels)