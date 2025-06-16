#!/usr/bin/env bash

action() {
    # This is for the HEAD@PR (including main merged)

    # setup output dir
    local current_git_hash
    current_git_hash=$(git rev-parse --verify HEAD)
    local results_dir=${BASE_OUTPUT_DIR}/${BRANCH_NAME}__${current_git_hash}
    local output_path_feature=${results_dir}/${bm_script}.json

    # Temporarily merge the target branch
    git checkout -b pr_branch
    git fetch --unshallow || echo "" # It might be worth switching actions/checkout to use depth 0 later on
    git config user.email "gha@example.com" && git config user.name "GHA" # For some reason this is needed even though nothing is being committed
    # shellcheck disable=SC2028
    git merge --no-commit --no-ff origin/"${TARGET_BRANCH}" || (echo "***\nError: There are merge conflicts that need to be resolved.\n***" && false)

    # create
    mkdir -p "$results_dir"

    local bm_script="benchmark.py"

    python $bm_script \
        --benchmark_time_unit=ms \
        --benchmark_out="${output_path_feature}" \
        --benchmark_out_format=json


    # This is for HEAD@main (usually main, not necessarily though)
    git stash
    git checkout origin/"${TARGET_BRANCH}"

    local current_git_hash
    current_git_hash=$(git rev-parse --verify HEAD)
    local results_dir=${BASE_OUTPUT_DIR}/${TARGET_BRANCH}__${current_git_hash}
    local output_path_target=${results_dir}/${bm_script}.json

    # create
    mkdir -p "$results_dir"

    local bm_script="benchmark.py"

    python $bm_script \
        --benchmark_time_unit=ms \
        --benchmark_out="${output_path_target}" \
        --benchmark_out_format=json

    # Compare both
    python compare.py "${output_path_target}" "${output_path_feature}"
}
action "$@"
