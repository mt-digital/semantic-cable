#! /bin/bash

for network in FOXNEWSW MSNBCW; do
    for n_weeks in 0; do
        for n_topics in 50 100 150 200 250 300 350 400 450 500; do
            qsub -v network=$network -v n_weeks=$n_weeks -v n_topics=$n_topics\
                -N "${network}_${n_weeks}_${n_topics}" \
                -o "${network}_${n_weeks}_${n_topics}.log" \
                ntopics_parameterization.sub
        done
    done
done
