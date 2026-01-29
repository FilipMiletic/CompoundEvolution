#!/bin/bash

# Path to output directory for coarse-grained models
# (will contain subdirectories for individual model runs)
out_coarse_main=

# Path to output directory for fine-grained models
# (will contain subdirectories for individual model runs)
out_fine_main=

# Path to directory containing decade-level one-sent-per-line corpus
corpus_dir=

# Path to log file
log_file=log_w2v_train.txt

for run in {1..5}; do

    # Output directories
    out_coarse=$out_coarse_main/run$run
    out_fine=$out_fine_main/run$run

    # W2V params
    dims=100
    win=10
    freq=1
    algo=sg

    # Coarse-grained training

    tmp_dir=$corpus_dir/tmp
    mkdir -p $tmp_dir

    for year1 in {1830..2000..30}; do

        year2=$((year1+10))
        year3=$((year1+20))

        # Move coarse grained data into temp directory for training
        for year in $year1 $year2 $year3; do
            corpus_file=$corpus_dir/cleaned_${year}s.txt.gz
            mv $corpus_file $tmp_dir
        done

        echo Checking contents of $tmp_dir: $(ls $tmp_dir) >> $log_file

        # Train coarse grained model
        python3 w2v_train.py $tmp_dir $out_coarse --dims $dims --win $win --freq $freq --algo $algo 2>> $log_file

        # Move coarse-grained data back into main directory
        mv $tmp_dir/* $corpus_dir

    done

    rm -df $tmp_dir

    # Fine-grained training

    for year in {1830..2000..10}; do
        corpus_file=$corpus_dir/cleaned_${year}s.txt.gz
        python3 w2v_train.py $corpus_file $out_fine --dims $dims --win $win --freq $freq --algo $algo 2>> $log_file
    done
    
done
