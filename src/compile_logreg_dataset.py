import pandas as pd
import random
import re


def main() -> None:

    # donors and loans used in training pair ngram lms
    trained = []
    with open("data/datasets/donor_loan_pairs_train.tsv", "r") as source:
        for line in source:
            eng, rus = line.strip().split('\t')
            trained.append(rus)

    test_donor_loans = []
    with open("data/datasets/donor_loan_pairs_test.tsv", "r") as source:
        for line in source:
            eng, rus = line.strip().split('\t')
            if rus not in test_donor_loans:
                test_donor_loans.append(rus)

    natives = []
    # Sholokhov Moscow Uni list of native Russian words
    native_source = pd.read_csv(
        "data/datasets/native_studfile.net.csv", header=None
    )
    scraped_natives = native_source[0].unique().tolist()
    for word in scraped_natives:
        word = re.sub("ё", "е", word)
        # skip words used in the pair n-gram dataset
        if word in trained or word in test_donor_loans:
            continue
        if " " in word:
            words = word.split(" ")
        else:
            words = [word]
        for w in words:
            w = w.lower()
            if w.lower not in natives:
                natives.append(w)
            if "-" in w:
                splits = w.split("-")
                for split in splits:
                    if len(split) > 1 and split not in natives:
                        natives.append(split)
    print("native rus words:", len(natives))

    # shuffle native
    random.seed(8)
    random.shuffle(natives)

    cutoff = round(len(natives) * 0.9)
    train_native = natives[:cutoff]
    test_native = natives[cutoff:]

    print("native train set:", len(train_native))
    print("native test set:", len(test_native))
    print("loan train set:", len(trained))
    print("loan test set:", len(test_donor_loans))
    print("total train set:", len(trained + train_native))
    print("total test set:", len(test_native + test_donor_loans))
    print("total dataset:", len(test_native + test_donor_loans + train_native + trained))

    with open("data/datasets/logreg_data.tsv", "w") as sink:
        for train in trained:
            print(f"{train}\t1", file=sink)
        for train_n in train_native:
            print(f"{train_n}\t0", file=sink)
        for test in test_donor_loans:
            print(f'{test}\t1', file=sink)
        for test_n in test_native:
            print(f'{test_n}\t0', file=sink)


if __name__ == "__main__":
    main()

    """
    native rus words: 3649
    native train set: 3284
    native test set: 365
    loan train set: 15358
    loan test set: 1706
    total train set: 18642
    total test set: 2071
    total dataset: 20713
    """
