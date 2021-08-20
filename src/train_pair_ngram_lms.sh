#!/bin/bash

NGRAMS=(3 4 5 6 7 8)

for n in ${NGRAMS[*]}; do
    ngramcount \
            --order="$n" \
            --require_symbols=false \
            ./enc/eng_rus_train.far | \
    ngrammake --method="kneser_ney" - | \
    ngramshrink \
            --method=relative_entropy \
            --target_number_of_ngrams=100000 \
            - | \
    fstencode --decode - ./enc/eng_rus_encoder_train.enc "./lms/eng_rus$n.fst"

done
