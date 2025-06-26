import string, re
import pandas as pd

remove = string.digits + string.punctuation
rows = []

with open("SMSSpamCollection.txt", "r") as input_file:
    for row in input_file:
        class_, text = row.split(maxsplit=1)

        cleaned_text = re.sub(rf"[{remove}]+", "", text).lower()
        words = list(filter(lambda s: len(s) > 1, re.findall(r"[a-z]+", cleaned_text)))

        rows.append((class_, words))

df = pd.DataFrame(rows, columns=["class", "words"])
print(df.head(5))

# Frequency of both ham and spam messages
print(df["class"].value_counts())

df.to_csv("SMSSpamCollection.csv", index=False)