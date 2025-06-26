from classifier import BayesClassifier
from clean_data import get_spam_collection_df
import pandas as pd

df = get_spam_collection_df()
df = df.sample(frac=1) # shuffled order

train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

model = BayesClassifier(["ham", "spam"])
model.train(train_df)

results = model.test(test_df)
print(results)
accuracy = results.loc[True] / results.sum()
print(f"Accuracy: {accuracy:.2%}")
