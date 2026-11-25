#!/usr/bin/env bash
# Every experiment of the report: the feature comparison, then C and the second stage.
# Usage: ./run_experiments.sh <dataset folder> [output folder]
# RUN can be set to another launcher; it is called as: $RUN train.py <args>
set -e
DATA="$1"
OUT="${2:-runs}"
RUN="${RUN:-python}"

run() {
    name="$1"
    shift
    "$RUN" train.py --data "$DATA" --out "$OUT" --name "$name" "$@"
    "$RUN" evaluate.py --data "$DATA" --run "$OUT/$name"
}

rerank() {
    "$RUN" train_reranker.py --data "$DATA" --run "$OUT/$1" --kind "$2"
    "$RUN" evaluate.py --data "$DATA" --run "$OUT/$1_$2"
}

run hog --features hog
run hog_lbp --features hog+lbp
run hog_color --features hog+color
run hog_lbp_color --features hog+lbp+color --keep-pool
run c1e3 --features hog+lbp+color --C 1e-3 --keep-pool

rerank hog_lbp_color rf
rerank hog_lbp_color gbdt
rerank c1e3 rf
rerank c1e3 gbdt  # the final model
