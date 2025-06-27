from classifier import BayesClassifier
from clean_data import split_train_test
import unittest

class BayesClassifierTest(unittest.TestCase):
    def setUp(self):
        self.train_df, self.test_df, self.class_labels = split_train_test()
        self.model = BayesClassifier(self.class_labels)
        self.model.train(self.train_df)
        self.confusion_matrix = self.model.confusion_matrix(self.test_df)

    def test_overall_accuracy(self):
        return self.assertGreaterEqual(self.model.get_accuracy(self.test_df), 0.8, "Model has bad overall accuracy.")

    def test_accuracy_by_class(self):
        for class_label in self.model.class_labels:
            with self.subTest(class_label=class_label):
                accuracy = self.model.get_accuracy_by_class(self.test_df, class_label)
                self.assertGreaterEqual(accuracy, 0.7, f"Model has bad accuracy for class '{class_label}'.")

if __name__ == "__main__":
    unittest.main()
