import csv
import string, re
import pandas as pd

remove = string.digits + string.punctuation

def get_document_collection_df():
    rows = []

    with open("bbc_data.csv", "r") as input_file:
        input_reader = csv.reader(input_file, delimiter=",", quotechar='"')
        for row in input_reader:
            text, label = row

            cleaned_text = re.sub(rf"[{remove}]+", "", text).lower()
            words = list(filter(lambda s: len(s) > 1, re.findall(r"[a-z]+", cleaned_text)))

            rows.append((label, words))

    return pd.DataFrame(rows, columns=["class", "words"])

def split_train_test(percent_train:float=0.8) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = get_document_collection_df().sample(frac=1)  # shuffled order

    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    class_labels = df["class"].unique().tolist()

    return train_df, test_df, class_labels

if __name__ == '__main__':
    df = get_document_collection_df()
    print(df.head(5))

    # Frequency of both ham and spam messages
    print(df["class"].value_counts())

    # test split train/test
    print(split_train_test())

