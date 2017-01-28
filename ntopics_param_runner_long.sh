#! /bin/bash

for network in FOXNEWSW MSNBCW; do
    # for n_weeks in 0 3 6 9 12; do
    for n_weeks in 0; do
        # for n_topics in 50 100 150 200 250 300 350 400 450 500; do
        for n_topics in 50 100 150 200 250 300 350 400 450 500; do
        # for n_topics in 300 350 400 450 500; do
            qsub -v network=$network -v n_weeks=$n_weeks -v n_topics=$n_topics\
                -N "long_${network}_${n_weeks}_${n_topics}" \
                -o "longlogs/${network}_${n_weeks}_${n_topics}.log" \
                ntopics_parameterization_long.sub
        done
    done
done
