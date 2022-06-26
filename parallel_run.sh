#!/bin/bash

# For HPOBench + rf/mlp
# Number of tasks (cpus) to run in parallel
N=20

for i in {0..100}; do
    (
        python lfbo_benchmark.py --benchmark fcnet_alt --dataset parkinsons --weight_type ei --model_type rf --iterations 200 --start_seed $i --end_seed $i
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
wait
