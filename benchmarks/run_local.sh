#!/usr/bin/env bash

action() {
    # setup output dir
    local current_git_hash=$(git rev-parse --verify HEAD)
    local results_dir=results/${current_git_hash}

    # create
    mkdir -p $results_dir

    local bm_script="benchmark.py"

    python $bm_script \
        --benchmark_time_unit=ms \
        --benchmark_color=true \
        --benchmark_out=$results_dir/$bm_script.json \
        --benchmark_out_format=json
}
action "$@"
