import random
import re


def main():
    pairs = []
    with open("data/datasets/scraped_donor_loan_pairs.tsv", 'r') as source:
        for line in source:
            donor, loan = line.strip().split("\t")
            # substitute ё
            loan = re.sub("ё", "е", loan)
            pair = f"{donor}_{loan}"
            if pair not in pairs:
                pairs.append(pair)

    random.seed(3)
    random.shuffle(pairs)

    train = pairs[:round(len(pairs) * 0.9)]
    test = pairs[round(len(pairs) * 0.9):]

    print("donor-loan train set:", len(train))  # 15358
    print("donor-loan test set", len(test))  # 1706

    for t in test:
        assert t not in train
    for r in train:
        assert r not in test

    with open("data/datasets/donor_loan_pairs_train.tsv", "w") as train_sink:
        for t in train:
            donor, loan = t.split("_")
            print(f"{donor}\t{loan}", file=train_sink)

    with open("data/datasets/donor_loan_pairs_test.tsv", "w") as test_sink:
        for t in test:
            donor, loan = t.split("_")
            print(f"{donor}\t{loan}", file=test_sink)


if __name__ == '__main__':
    main()
