#!/bin/bash

# Path to output directory in which to write feature information
out_dir=

# Path to log file
log_file=

# Path to directory with trained fine-grained models
# (expecting multiple training runs in subdirectories)
fine_dir_main=

# Path to directory with trained coarse-grained models
# (expecting multiple training runs in subdirectories)
coarse_dir_main=

for run in {1..5}; do

    fine_dir=$fine_dir_main/run$run
    coarse_dir=$coarse_dir_main/run$run
    
    python w2v_synchronic_cos.py $fine_dir $coarse_dir $out_dir 2>> $log_file
    python w2v_diachronic_cos.py $fine_dir $coarse_dir $out_dir 2>> $log_file

    for k in 10 20 50 100 200 500 1000; do
        python w2v_synchronic_neighb.py $fine_dir $coarse_dir $out_dir -k $k 2>> $log_file
        python w2v_diachronic_neighb.py $fine_dir $coarse_dir $out_dir -k $k 2>> $log_file
    done
    
done