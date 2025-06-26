from classifier import BayesClassifier
from clean_data import split_train_test
import pandas as pd

def get_accuracy(model: BayesClassifier, test_df: pd.DataFrame) -> float:
    results = model.test(test_df, accuracy_only=True)
    accuracy = results.loc[True] / results.sum()
    return accuracy

if __name__ == "__main__":
    train_df, test_df, class_labels = split_train_test()
    model = BayesClassifier(class_labels)
    model.train(train_df)
    print(model.class_probs)
    print(f"Accuracy: {get_accuracy(model, test_df):0.2%}")

