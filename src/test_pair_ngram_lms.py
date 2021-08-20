#!/usr/bin/env python

import csv
import logging
import nltk
import os
import pandas as pd

import pynini


def main(lm_directory: str, test_file: str, results_save_path: str) -> None:

    lms = [file for file in os.listdir(lm_directory) if 'rus_eng' in file]

    donors = []
    loans = []
    with open(test_file, 'r') as source:
        for line in source:
            donor, loan = line.strip().split("\t")
            loans.append(loan.strip())
            donors.append(donor.strip())

    test_result_columns = ['lm', 'cost', 'wer']
    with open(results_save_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=test_result_columns)
        writer.writeheader()

        # get cost and WER per lm
        for lm in lms:
            lm_path = lm_directory + lm
            fst = pynini.Fst.read(lm_path)
            lm_wers = []
            lm_costs = []

            # get predictions
            for i, loan in enumerate(loans):
                acceptor = pynini.accep(loan, token_type='utf8')
                lattice = acceptor @ fst
                predicted_donor = ''
                probability = 0.0
                if lattice.start() != pynini.NO_STATE_ID:
                    lattice.project("output")
                    lattice = pynini.shortestpath(lattice, unique=True)
                    cost = pynini.shortestdistance(lattice, reverse=True)[lattice.start()]
                    probability = float(cost)
                    predicted_donor = pynini.shortestpath(lattice).string()
                else:
                    print(f"No lattice start for {loan} by {lm}")
                edit_distance = nltk.edit_distance(predicted_donor, donors[i])
                lm_wers.append(edit_distance)
                lm_costs.append(probability)

            average_cost = sum(lm_costs) / len(loans)
            average_wer = sum(lm_wers) / len(loans)
            lm_results = {
                'lm': lm,
                "cost": average_cost,
                "wer": average_wer
            }
            writer.writerow(lm_results)
            logging.info(f"{lm} tested: cost: {average_cost}, wer: {average_wer}")


if __name__ == '__main__':
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    directory = "data/lms/"
    test_file = "data/datasets/donor_loan_pairs_test.tsv"
    test_results_csv = 'results/pair_ngram_lms_test_results.csv'
    main(directory, test_file, test_results_csv)
