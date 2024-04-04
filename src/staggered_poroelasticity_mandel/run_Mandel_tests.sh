#!/bin/bash

# This script runs the staggered multirate method for a lot of different configurations
# and saves the results in the folder "results/Mandel/""

# create a counter
counter=1

for problem in "Mandel"
do
    echo "Running problem $problem"
    for criterion in "residual" "solution"
    do
        echo "  Using convergence criterion $criterion"
        for n_time_p in {1,2,4,8,16,32,64,128,256}
        do

            # print counter
            echo "    $counter: Using n_time_p = $n_time_p"

            # run the code and save the output in the folder
            start_time=`date +%s`
            python3 Mandel_multirate_staggered.py --n_time_p $n_time_p --convergence_criterion $criterion 
            end_time=`date +%s`
	        echo "      TIME: Running the code took $((end_time-start_time)) seconds."
            ((counter++))
        done
    done
done