from classifier import BayesClassifier
from clean_data import split_train_test
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

train, test, class_labels = split_train_test()
model = BayesClassifier(class_labels)
model.train(train)

@app.route("/")
def home():
    evals = model.test(test).rename("evals")
    test_classes = test.drop(columns=["words"]).reset_index(drop=True)

    merged = pd.merge(evals, test_classes, left_index=True, right_index=True)
    merged["predicted"] = merged["evals"] == merged["class"]

    return render_template("index.html", model=model, train=train, test=test, evals=evals, merged=merged)