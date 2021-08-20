#!/bin/bash

python src/donor_loan_data_split.py
python src/compile_logreg_dataset.py
source src/align_pair_ngrams.sh
source src/train_pair_ngram_lms.sh
python src/invert_fst.py
python src/test_pair_ngram_lms.py
python src/train_logreg.py \
        --data_path=data/datasets/logreg_data.tsv \
        --eng_lexicon_path=data/datasets/english_lexicon_dict.json \
        --results_path=results/logreg_test_results.csv \
        --fsts_dir=data/lms/