#!/bin/bash

# using  https://github.com/sigmorphon/2020/blob/master/task1/baselines/fst/align.py

python3 align.py \
        --tsv_path=./data/donor_loan_pairs_train.tsv \
        --far_path=./enc/eng_rus_train.far \
        --encoder_path=./enc/eng_rus_encoder_train.enc \
        --random_starts=12 \
        --seed=13

