from classifier import BayesClassifier
from clean_data import split_train_test
from flask import Flask, render_template
from test import get_accuracy
import pandas as pd

app = Flask(__name__)

model = BayesClassifier(["ham", "spam"])
train, test = split_train_test()
model.train(train)

@app.route("/")
def home():
    global test
    evals = model.test(test).rename("evals")
    test_classes = test.drop(columns=["words"]).reset_index(drop=True)

    merged = pd.merge(evals, test_classes, left_index=True, right_index=True)
    merged["predicted"] = merged["evals"] == merged["class"]
    print(merged.value_counts())
    print(get_accuracy(model, test) * 1115)

    return render_template("index.html", model=model, train=train, test=test, evals=evals)